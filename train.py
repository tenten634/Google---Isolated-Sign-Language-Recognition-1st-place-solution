#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for Google Isolated Sign Language Recognition 1st place solution
Converted from notebook to Python script with GPU support and logging integration
"""

import os
import sys
import gc
import re
import glob
import time
import math
import random
import pickle
import argparse
import datetime
from copy import copy

import numpy as np
import pandas as pd
import tensorflow as tf
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
except ImportError:
    TFA_AVAILABLE = False
    print("⚠️  TensorFlow Addons not available - falling back to AdamW")
import tensorflow.keras.mixed_precision as mixed_precision
from tqdm import tqdm

# Try to import tf_utils (custom package)
try:
    sys.path.append('tf_utils')
    from tf_utils.schedules import OneCycleLR, ListedLR
    from tf_utils.callbacks import Snapshot, SWA
    from tf_utils.learners import FGM, AWP
    TF_UTILS_AVAILABLE = True
except Exception as e:
    print(f"⚠️  tf_utils not available - falling back to standard schedules ({e})")
    TF_UTILS_AVAILABLE = False

# Initialize wandb (optional)
wandb_run = None

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Wandb not available - wandb logging will be disabled")



# Constants
ROWS_PER_FRAME = 543
MAX_LEN = 384
CROP_LEN = MAX_LEN
NUM_CLASSES = 250
PAD = -100.

# Landmark indices
NOSE = [1, 2, 98, 327]
LNOSE = [98]
RNOSE = [327]
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]
POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513, 505, 503, 501]
RPOSE = [512, 504, 502, 500]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = len(POINT_LANDMARKS)
CHANNELS = 6 * NUM_NODES


def seed_everything(seed=42):
    """Seed all random number generators."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_strategy(device='GPU'):
    """
    Get TensorFlow distribution strategy.
    Modified to use GPU instead of TPU.
    """
    IS_TPU = False
    
    if device == "GPU" or device == "CPU":
        ngpu = len(tf.config.experimental.list_physical_devices('GPU'))
        if ngpu > 1:
            print(f"Using {ngpu} GPUs with MirroredStrategy")
            strategy = tf.distribute.MirroredStrategy()
        elif ngpu == 1:
            print("Using single GPU")
            strategy = tf.distribute.get_strategy()
        else:
            print("Using CPU")
            strategy = tf.distribute.get_strategy()
            device = "CPU"
    
    if device == "GPU":
        print(f"Num GPUs Available: {ngpu}")
    
    AUTO = tf.data.experimental.AUTOTUNE
    REPLICAS = strategy.num_replicas_in_sync
    print(f'REPLICAS: {REPLICAS}')
    
    return strategy, REPLICAS, IS_TPU


def count_data_items(filenames):
    """Count data items from TFRecord filenames."""
    n = [int(re.compile(r"-([0-9]*)\.").search(filename.split('/')[-1]).group(1)) 
         for filename in filenames]
    return np.sum(n)


def interp1d_(x, target_len, method='random'):
    """Interpolate 1D tensor."""
    length = tf.shape(x)[1]
    target_len = tf.maximum(1, target_len)
    if method == 'random':
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bilinear')
        else:
            if tf.random.uniform(()) < 0.5:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bicubic')
            else:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'nearest')
    else:
        x = tf.image.resize(x, (target_len, tf.shape(x)[1]), method)
    return x


def tf_nan_mean(x, axis=0, keepdims=False):
    """Compute mean ignoring NaN values."""
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / \
           tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)


def tf_nan_std(x, center=None, axis=0, keepdims=False):
    """Compute std ignoring NaN values."""
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))


class Preprocess(tf.keras.layers.Layer):
    """Preprocessing layer for landmark data."""
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        is_rank3 = tf.equal(tf.rank(inputs), 3)
        x = tf.cond(is_rank3, lambda: inputs[None, ...], lambda: inputs)
        
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        
        x = (x - mean) / std
        
        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]
        
        dx = tf.cond(tf.shape(x)[1] > 1, 
                    lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]), 
                    lambda: tf.zeros_like(x))
        
        dx2 = tf.cond(tf.shape(x)[1] > 2,
                     lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
                     lambda: tf.zeros_like(x))
        
        x = tf.concat([
            tf.reshape(x, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
        ], axis=-1)
        
        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        
        return x


def decode_tfrec(record_bytes):
    """Decode TFRecord."""
    features = tf.io.parse_single_example(record_bytes, {
        'coordinates': tf.io.FixedLenFeature([], tf.string),
        'sign': tf.io.FixedLenFeature([], tf.int64),
    })
    out = {}
    out['coordinates'] = tf.reshape(tf.io.decode_raw(features['coordinates'], tf.float32), (-1, ROWS_PER_FRAME, 3))
    out['sign'] = features['sign']
    return out


def filter_nans_tf(x, ref_point=POINT_LANDMARKS):
    """Filter NaN values from tensor."""
    mask = tf.math.logical_not(tf.reduce_all(tf.math.is_nan(tf.gather(x, ref_point, axis=1)), axis=[-2, -1]))
    x = tf.boolean_mask(x, mask, axis=0)
    return x


def flip_lr(x):
    """Flip left-right augmentation."""
    x, y, z = tf.unstack(x, axis=-1)
    x = 1 - x
    new_x = tf.stack([x, y, z], -1)
    new_x = tf.transpose(new_x, [1, 0, 2])
    
    lhand = tf.gather(new_x, LHAND, axis=0)
    rhand = tf.gather(new_x, RHAND, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LHAND)[..., None], rhand)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RHAND)[..., None], lhand)
    
    llip = tf.gather(new_x, LLIP, axis=0)
    rlip = tf.gather(new_x, RLIP, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LLIP)[..., None], rlip)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RLIP)[..., None], llip)
    
    lpose = tf.gather(new_x, LPOSE, axis=0)
    rpose = tf.gather(new_x, RPOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LPOSE)[..., None], rpose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RPOSE)[..., None], lpose)
    
    leye = tf.gather(new_x, LEYE, axis=0)
    reye = tf.gather(new_x, REYE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LEYE)[..., None], reye)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(REYE)[..., None], leye)
    
    lnose = tf.gather(new_x, LNOSE, axis=0)
    rnose = tf.gather(new_x, RNOSE, axis=0)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(LNOSE)[..., None], rnose)
    new_x = tf.tensor_scatter_nd_update(new_x, tf.constant(RNOSE)[..., None], lnose)
    new_x = tf.transpose(new_x, [1, 0, 2])
    return new_x


def resample(x, rate=(0.8, 1.2)):
    """Resample augmentation."""
    rate = tf.random.uniform((), rate[0], rate[1])
    length = tf.shape(x)[0]
    new_size = tf.cast(rate * tf.cast(length, tf.float32), tf.int32)
    new_x = interp1d_(x, new_size)
    return new_x


def spatial_random_affine(xyz, scale=(0.8, 1.2), shear=(-0.15, 0.15), shift=(-0.1, 0.1), degree=(-30, 30)):
    """Spatial random affine augmentation."""
    center = tf.constant([0.5, 0.5])
    if scale is not None:
        scale = tf.random.uniform((), *scale)
        xyz = scale * xyz
    
    if shear is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        shear_x = shear_y = tf.random.uniform((), *shear)
        if tf.random.uniform(()) < 0.5:
            shear_x = 0.
        else:
            shear_y = 0.
        shear_mat = tf.identity([[1., shear_x], [shear_y, 1.]])
        xy = xy @ shear_mat
        center = center + [shear_y, shear_x]
        xyz = tf.concat([xy, z], axis=-1)
    
    if degree is not None:
        xy = xyz[..., :2]
        z = xyz[..., 2:]
        xy -= center
        degree = tf.random.uniform((), *degree)
        radian = degree / 180 * np.pi
        c = tf.math.cos(radian)
        s = tf.math.sin(radian)
        rotate_mat = tf.identity([[c, s], [-s, c]])
        xy = xy @ rotate_mat
        xy = xy + center
        xyz = tf.concat([xy, z], axis=-1)
    
    if shift is not None:
        shift = tf.random.uniform((), *shift)
        xyz = xyz + shift
    
    return xyz


def temporal_crop(x, length=MAX_LEN):
    """Temporal crop augmentation."""
    l = tf.shape(x)[0]
    offset = tf.random.uniform((), 0, tf.clip_by_value(l - length, 1, length), dtype=tf.int32)
    x = x[offset:offset + length]
    return x


def temporal_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    """Temporal mask augmentation."""
    l = tf.shape(x)[0]
    mask_size = tf.random.uniform((), *size)
    mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
    mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l - mask_size, 1, l), dtype=tf.int32)
    x = tf.tensor_scatter_nd_update(x, tf.range(mask_offset, mask_offset + mask_size)[..., None],
                                    tf.fill([mask_size, 543, 3], mask_value))
    return x


def spatial_mask(x, size=(0.2, 0.4), mask_value=float('nan')):
    """Spatial mask augmentation."""
    mask_offset_y = tf.random.uniform(())
    mask_offset_x = tf.random.uniform(())
    mask_size = tf.random.uniform((), *size)
    mask_x = (mask_offset_x < x[..., 0]) & (x[..., 0] < mask_offset_x + mask_size)
    mask_y = (mask_offset_y < x[..., 1]) & (x[..., 1] < mask_offset_y + mask_size)
    mask = mask_x & mask_y
    x = tf.where(mask[..., None], mask_value, x)
    return x


def augment_fn(x, always=False, max_len=None):
    """Apply augmentations."""
    if tf.random.uniform(()) < 0.8 or always:
        x = resample(x, (0.5, 1.5))
    if tf.random.uniform(()) < 0.5 or always:
        x = flip_lr(x)
    if max_len is not None:
        x = temporal_crop(x, max_len)
    if tf.random.uniform(()) < 0.75 or always:
        x = spatial_random_affine(x)
    if tf.random.uniform(()) < 0.5 or always:
        x = temporal_mask(x)
    if tf.random.uniform(()) < 0.5 or always:
        x = spatial_mask(x)
    return x


def preprocess(x, augment=False, max_len=MAX_LEN):
    """Preprocess data."""
    coord = x['coordinates']
    coord = filter_nans_tf(coord)
    if augment:
        coord = augment_fn(coord, max_len=max_len)
    coord = tf.ensure_shape(coord, (None, ROWS_PER_FRAME, 3))
    
    return tf.cast(Preprocess(max_len=max_len)(coord)[0], tf.float32), tf.one_hot(x['sign'], NUM_CLASSES)


def get_tfrec_dataset(tfrecords, batch_size=64, max_len=64, drop_remainder=False, augment=False, shuffle=False, repeat=False):
    """Create TFRecord dataset."""
    ds = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE, compression_type='GZIP')
    ds = ds.map(decode_tfrec, tf.data.AUTOTUNE)
    ds = ds.map(lambda x: preprocess(x, augment=augment, max_len=max_len), tf.data.AUTOTUNE)
    
    if repeat:
        ds = ds.repeat()
    
    if shuffle:
        ds = ds.shuffle(shuffle)
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = ds.with_options(options)
    
    if batch_size:
        ds = ds.padded_batch(batch_size, padding_values=PAD, 
                           padded_shapes=([max_len, CHANNELS], [NUM_CLASSES]), 
                           drop_remainder=drop_remainder)
    
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds


class ECA(tf.keras.layers.Layer):
    """Efficient Channel Attention layer."""
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn


class LateDropout(tf.keras.layers.Layer):
    """Late dropout layer."""
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)
      
    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(self._train_counter < self.start_step, 
                   lambda: inputs, 
                   lambda: self.dropout(inputs, training=training))
        if training:
            self._train_counter.assign_add(1)
        return x


class CausalDWConv1D(tf.keras.layers.Layer):
    """Causal depthwise 1D convolution."""
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False,
                 depthwise_initializer='glorot_uniform', name='', **kwargs):
        super().__init__(name=name, **kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0), name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size, strides=1, dilation_rate=dilation_rate,
            padding='valid', use_bias=use_bias,
            depthwise_initializer=depthwise_initializer, name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x


def Conv1DBlock(channel_size, kernel_size, dilation_rate=1, drop_rate=0.0,
                expand_ratio=2, se_ratio=0.25, activation='swish', name=None):
    """Efficient conv1d block."""
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
    
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio
        skip = inputs
        
        x = tf.keras.layers.Dense(channels_expand, use_bias=True, activation=activation,
                                 name=name + '_expand_conv')(inputs)
        x = CausalDWConv1D(kernel_size, dilation_rate=dilation_rate, use_bias=False,
                          name=name + '_dwconv')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)
        x = ECA()(x)
        x = tf.keras.layers.Dense(channel_size, use_bias=True, name=name + '_project_conv')(x)
        
        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + '_drop')(x)
        
        if channels_in == channel_size:
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x
    
    return apply


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention layer."""
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
        
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        
        if mask is not None:
            mask = mask[:, None, None, :]
        
        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)
        
        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    """Transformer block."""
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x
        
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply


def get_model(max_len=64, dropout_step=0, dim=192):
    """Build model."""
    inp = tf.keras.Input((max_len, CHANNELS))
    x = tf.keras.layers.Masking(mask_value=PAD, input_shape=(max_len, CHANNELS))(inp)
    ksize = 17
    x = tf.keras.layers.Dense(dim, use_bias=False, name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95, name='stem_bn')(x)
    
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)
    
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
    x = TransformerBlock(dim, expand=2)(x)
    
    if dim == 384:  # for the 4x sized model
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)
        
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)
        x = TransformerBlock(dim, expand=2)(x)
    
    x = tf.keras.layers.Dense(dim * 2, activation=None, name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, name='classifier')(x)
    return tf.keras.Model(inp, x)


class LoggingCallback(tf.keras.callbacks.Callback):
    """Custom callback for logging to wandb."""
    def __init__(self, wandb_run=None):
        super().__init__()
        self.wandb_run = wandb_run
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        step = epoch
        
        for key, value in logs.items():
            if self.wandb_run:
                self.wandb_run.log({key: value}, step=step)


def train_fold(CFG, fold, train_files, valid_files=None, strategy=None, summary=True,
               wandb_run=None):
    """Train a single fold."""
    seed_everything(CFG.seed)
    tf.keras.backend.clear_session()
    gc.collect()
    tf.config.optimizer.set_jit(True)
    
    if CFG.fp16:
        try:
            policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_global_policy(policy)
        except:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
    else:
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_global_policy(policy)
    
    if fold != 'all':
        train_ds = get_tfrec_dataset(train_files, batch_size=CFG.batch_size, max_len=CFG.max_len,
                                    drop_remainder=True, augment=True, repeat=True, shuffle=32768)
        valid_ds = get_tfrec_dataset(valid_files, batch_size=CFG.batch_size, max_len=CFG.max_len,
                                    drop_remainder=False, repeat=False, shuffle=False)
    else:
        train_ds = get_tfrec_dataset(train_files, batch_size=CFG.batch_size, max_len=CFG.max_len,
                                     drop_remainder=False, augment=True, repeat=True, shuffle=32768)
        valid_ds = None
        valid_files = []
    
    num_train = count_data_items(train_files)
    num_valid = count_data_items(valid_files)
    steps_per_epoch = num_train // CFG.batch_size
    
    with strategy.scope():
        dropout_step = CFG.dropout_start_epoch * steps_per_epoch
        model = get_model(max_len=CFG.max_len, dropout_step=dropout_step, dim=CFG.dim)
        
        if TF_UTILS_AVAILABLE:
            schedule = OneCycleLR(
                CFG.lr,
                CFG.epoch,
                warmup_epochs=CFG.epoch * CFG.warmup,
                steps_per_epoch=steps_per_epoch,
                resume_epoch=CFG.resume,
                decay_epochs=CFG.epoch,
                lr_min=CFG.lr_min,
                decay_type=CFG.decay_type,
                warmup_type='linear',
            )
            decay_schedule = OneCycleLR(
                CFG.lr * CFG.weight_decay,
                CFG.epoch,
                warmup_epochs=CFG.epoch * CFG.warmup,
                steps_per_epoch=steps_per_epoch,
                resume_epoch=CFG.resume,
                decay_epochs=CFG.epoch,
                lr_min=CFG.lr_min * CFG.weight_decay,
                decay_type=CFG.decay_type,
                warmup_type='linear',
            )
            
            awp_step = CFG.awp_start_epoch * steps_per_epoch
            if CFG.fgm:
                model = FGM(model.input, model.output, delta=CFG.awp_lambda, eps=0., start_step=awp_step)
            elif CFG.awp:
                model = AWP(model.input, model.output, delta=CFG.awp_lambda, eps=0., start_step=awp_step)
        else:
            total_steps = steps_per_epoch * CFG.epoch
            lr_min_ratio = CFG.lr_min / CFG.lr if CFG.lr > 0 else 0.0
            schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=CFG.lr,
                decay_steps=total_steps,
                alpha=lr_min_ratio,
            )
            decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=CFG.lr * CFG.weight_decay,
                decay_steps=total_steps,
                alpha=lr_min_ratio,
            )
        
        if TFA_AVAILABLE:
            opt = tfa.optimizers.RectifiedAdam(
                learning_rate=schedule,
                weight_decay=decay_schedule,
                sma_threshold=4,
            )
            opt = tfa.optimizers.Lookahead(opt, sync_period=5)
        else:
            opt = tf.keras.optimizers.AdamW(
                learning_rate=schedule,
                weight_decay=CFG.weight_decay,
            )
        
        model.compile(
            optimizer=opt,
            loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)],
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
            steps_per_execution=steps_per_epoch,
        )
    
    if summary:
        print()
        model.summary()
        print()
        print(train_ds, valid_ds)
        print()
    
    print(f'---------fold{fold}---------')
    print(f'train:{num_train} valid:{num_valid}')
    print()
    
    if CFG.resume:
        print(f'resume from epoch{CFG.resume}')
        model.load_weights(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-last.h5')
        if train_ds is not None:
            model.evaluate(train_ds.take(steps_per_epoch))
        if valid_ds is not None:
            model.evaluate(valid_ds)
    
    callbacks = []
    
    # Add logging callback
    logging_callback = LoggingCallback(wandb_run=wandb_run)
    callbacks.append(logging_callback)
    
    if CFG.save_output:
        logger = tf.keras.callbacks.CSVLogger(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-logs.csv')
        sv_loss = tf.keras.callbacks.ModelCheckpoint(
            f'{CFG.output_dir}/{CFG.comment}-fold{fold}-best.h5',
            monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')
        callbacks.append(logger)
        if TF_UTILS_AVAILABLE:
            snap = Snapshot(f'{CFG.output_dir}/{CFG.comment}-fold{fold}', CFG.snapshot_epochs)
            swa = SWA(
                f'{CFG.output_dir}/{CFG.comment}-fold{fold}',
                CFG.swa_epochs,
                strategy=strategy,
                train_ds=train_ds,
                valid_ds=valid_ds,
                valid_steps=-(num_valid // -CFG.batch_size),
            )
            callbacks.append(snap)
            callbacks.append(swa)
        if fold != 'all':
            callbacks.append(sv_loss)
    
    history = model.fit(
        train_ds,
        epochs=CFG.epoch - CFG.resume,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=valid_ds,
        verbose=CFG.verbose,
        validation_steps=-(num_valid // -CFG.batch_size)
    )
    
    if CFG.save_output:
        try:
            model.load_weights(f'{CFG.output_dir}/{CFG.comment}-fold{fold}-best.h5')
        except:
            pass
    
    if fold != 'all':
        cv = model.evaluate(valid_ds, verbose=CFG.verbose, steps=-(num_valid // -CFG.batch_size))
    else:
        cv = None
    
    return model, cv, history


def train_folds(CFG, folds, strategy=None, summary=True,
                wandb_run=None):
    """Train multiple folds."""
    for fold in folds:
        if fold != 'all':
            all_files = CFG.train_filenames
            train_files = [x for x in all_files if f'fold{fold}' not in x]
            valid_files = [x for x in all_files if f'fold{fold}' in x]
        else:
            train_files = CFG.train_filenames
            valid_files = None
        
        train_fold(CFG, fold, train_files, valid_files, strategy=strategy, summary=summary,
                  wandb_run=wandb_run)
    return


def main():
    parser = argparse.ArgumentParser(description="Train ISLR model")
    parser.add_argument("--data_dir", type=str, default="./datammount",
                       help="Base data directory for downloaded datasets")
    parser.add_argument("--tfrecords_subdir", type=str, default="ISLR-5fold",
                       help="Subdirectory under data_dir containing TFRecords")
    parser.add_argument("--competition_subdir", type=str, default="Google-Isolated-Sign-Language-Recognition",
                       help="Subdirectory under data_dir containing competition files")
    parser.add_argument("--train_filenames", type=str, default=None,
                       help="Glob pattern for TFRecord files (e.g., 'data/*.tfrecords')")
    parser.add_argument("--train_csv", type=str, default=None,
                       help="Path to train.csv")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Output directory for checkpoints")
    parser.add_argument("--fold", type=int, default=0,
                       help="Fold number (use -1 for 'all')")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--epoch", type=int, default=300,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (will be multiplied by num_gpus if not specified)")
    parser.add_argument("--dim", type=int, default=192,
                       help="Model dimension")
    parser.add_argument("--device", type=str, default="GPU",
                       choices=["GPU", "CPU"],
                       help="Device to use")
    parser.add_argument("--loggers", type=str, nargs="+", default=["wandb"],
                       choices=["wandb"],
                       help="Loggers to use")
    parser.add_argument("--wandb_project", type=str, default="isl-1",
                       help="Wandb project name")
    
    args = parser.parse_args()

    # Initialize logger handles for this run
    wandb_run = None
    
    # Get strategy
    strategy, N_REPLICAS, IS_TPU = get_strategy(device=args.device)
    
    # Adjust batch size and learning rate based on number of replicas
    batch_size = args.batch_size * N_REPLICAS
    lr = args.lr if args.lr is not None else 5e-4 * N_REPLICAS
    
    # Resolve default data paths if not explicitly provided
    if not args.train_filenames:
        tfrecords_dir = os.path.join(args.data_dir, args.tfrecords_subdir)
        args.train_filenames = os.path.join(tfrecords_dir, "*.tfrecords")
    
    if not args.train_csv:
        competition_dir = os.path.join(args.data_dir, args.competition_subdir)
        args.train_csv = os.path.join(competition_dir, "train.csv")
    
    # Get train filenames
    train_filenames = glob.glob(args.train_filenames)
    print(f"Found {len(train_filenames)} TFRecord files")
    if not train_filenames:
        raise FileNotFoundError(f"No TFRecords found with pattern: {args.train_filenames}")
    
    # Read train CSV
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"train.csv not found: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    print(f"Train DataFrame shape: {train_df.shape}")
    
    # Verify data count matches
    num_data_items = count_data_items(train_filenames)
    assert num_data_items == len(train_df), f"Data count mismatch: {num_data_items} vs {len(train_df)}"
    
    # Configuration
    class CFG:
        pass
    
    CFG.n_splits = 5
    CFG.save_output = True
    CFG.output_dir = args.output_dir
    CFG.train_filenames = train_filenames
    
    CFG.seed = args.seed
    CFG.verbose = 2
    
    CFG.max_len = MAX_LEN
    CFG.replicas = N_REPLICAS
    CFG.lr = lr
    CFG.weight_decay = 0.1
    CFG.lr_min = 1e-6
    CFG.epoch = args.epoch
    CFG.warmup = 0
    CFG.batch_size = batch_size
    CFG.snapshot_epochs = []
    CFG.swa_epochs = []
    
    CFG.fp16 = True
    CFG.fgm = False
    CFG.awp = True
    CFG.awp_lambda = 0.2
    CFG.awp_start_epoch = 15
    CFG.dropout_start_epoch = 15
    CFG.resume = 0
    CFG.decay_type = 'cosine'
    CFG.dim = args.dim
    CFG.comment = f'islr-fp16-{args.dim}-{N_REPLICAS}-seed{args.seed}'

    # Keras 2.x compatibility (no special handling needed)
    
    # Create output directory
    os.makedirs(CFG.output_dir, exist_ok=True)
    
    # Initialize loggers
    loggers_to_use = args.loggers
    
    # Build a clean config dict for logging (avoid non-serializable class attrs)
    cfg_keys = [
        "n_splits",
        "save_output",
        "output_dir",
        "train_filenames",
        "seed",
        "verbose",
        "max_len",
        "replicas",
        "lr",
        "weight_decay",
        "lr_min",
        "epoch",
        "warmup",
        "batch_size",
        "snapshot_epochs",
        "swa_epochs",
        "fp16",
        "fgm",
        "awp",
        "awp_lambda",
        "awp_start_epoch",
        "dropout_start_epoch",
        "resume",
        "decay_type",
        "dim",
        "comment",
    ]
    cfg_dict = {key: getattr(CFG, key) for key in cfg_keys}
    
    # Initialize wandb
    if 'wandb' in loggers_to_use and WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                tags=[f"fold{args.fold}", f"seed{args.seed}"],
                config=cfg_dict,
                mode=os.environ.get('WANDB_MODE', 'online')
            )
            print(f"✅ Wandb initialized: {wandb_run.name}")
            print(f"   URL: {wandb_run.url}")
        except Exception as e:
            print(f"⚠️  Wandb initialization failed: {e}")
            wandb_run = None
    
    # Train
    folds = [args.fold] if args.fold != -1 else ['all']
    train_folds(CFG, folds, strategy=strategy, summary=True,
               wandb_run=wandb_run)
    
    # Finish loggers
    if wandb_run:
        wandb_run.finish()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
