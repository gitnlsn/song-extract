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

    // Converta as labels para tensor 1D (não use oneHot para binário)
    const xs = tf.tensor2d(data); // shape: [num_samples, inputSize]
    const ys = tf.tensor1d(labels, 'float32'); // labels devem ser 0 ou 1

    // Crie o modelo
    const model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [inputSize],
                units: 128,
                activation: 'gelu',
                kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), // Regularização L2,
                kernelInitializer: 'heNormal'
            }),
            tf.layers.batchNormalization(), // Normalização para acelerar treinamento
            tf.layers.dense({ units: 32, activation: 'gelu', kernelInitializer: 'heNormal' }),
            tf.layers.dense({ units: 8, activation: 'gelu', kernelInitializer: 'heNormal' }),
            tf.layers.dense({ units: 1, activation: 'sigmoid', kernelInitializer: 'glorotNormal' }) // Única saída binária
        ]
    });

    // Compile o modelo
    model.compile({
        optimizer: 'adam', // Melhor que SGD
        loss: 'binaryCrossentropy',
        metrics: ['binaryAccuracy', 'precision']
    });

    // Treine o modelo
    await model.fit(xs, ys, {
        epochs: props?.epochs ?? 5,
        batchSize: props?.batchSize ?? 32,
        callbacks: {
            onBatchEnd: (batch, logs) => {
                console.log('Logs', logs);
            }
        },
        validationSplit: 0.2 // Separa 20% para validação
    });

    // Clean up tensors
    xs.dispose();
    ys.dispose();

    return model;
}
