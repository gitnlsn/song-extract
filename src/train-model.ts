import * as tf from '@tensorflow/tfjs';
import { AudioSegment } from './interfaces/AudioSegment';
import { TrainInput } from './interfaces/TrainInput';
import { Tensor1DInput, Tensor2DInput } from './interfaces/TensorFlowData';

interface TrainModelProps {
    data: Tensor2DInput;
    labels: Tensor1DInput;
    props?: { epochs?: number, batchSize?: number, inputSize?: number }
}

export async function trainModel({ data, labels, props }: TrainModelProps): Promise<tf.LayersModel> {
    const inputSize = props?.inputSize ?? 1000;
    
    // Convert to tensors
    const xs = tf.tensor2d(data);
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 2);

    // Create model
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [inputSize], units: 32, activation: 'gelu' }),
            tf.layers.dense({ units: 2, activation: 'softmax' }),
        ]
    });

    // Compile model
    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // Train model
    await model.fit(xs, ys, {
        epochs: props?.epochs ?? 5,
        batchSize: props?.batchSize ?? 32,
        callbacks: {
            onBatchEnd: (batch, logs) => {
                console.log('Accuracy', logs?.acc);
            }
        }
    });

    // Clean up tensors
    xs.dispose();
    ys.dispose();

    return model;
}
