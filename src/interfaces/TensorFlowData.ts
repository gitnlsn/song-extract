import * as tf from '@tensorflow/tfjs';

export type Tensor2DInput = Parameters<typeof tf.tensor2d>[0];
export type Tensor1DInput = Parameters<typeof tf.tensor1d>[0];