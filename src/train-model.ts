import * as tf from '@tensorflow/tfjs';
import { AudioSegment } from './interfaces/AudioSegment';
import { TrainInput } from './interfaces/TrainInput';

function isInSongInterval(segment: AudioSegment, intervals: TrainInput['songIntervals']): boolean {
    return intervals.some(interval =>
        segment.startTime >= interval.start && segment.endTime <= interval.end
    );
}

export async function trainModel(segments: AudioSegment[], trainInput: TrainInput, props?: { epochs?: number, batchSize?: number, inputSize?: number }): Promise<tf.LayersModel> {
    const inputSize = props?.inputSize ?? 1000;
    // Prepare data and labels
    const data: number[][] = [];
    const labels: number[] = [];

    segments.forEach(segment => {
        // Create a new array with the maximum length
        const paddedData = new Float32Array(inputSize);
        // Copy the original data
        paddedData.set(segment.decoded);
        // Fill the rest with zeros if needed
        data.push(Array.from(paddedData));

        // Set label based on whether the segment is within song intervals
        const isSong = isInSongInterval(segment, trainInput.songIntervals);
        labels.push(isSong ? 1 : 0);
    });

    // Convert to tensors
    const xs = tf.tensor2d(data);
    const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 2);

    // Create model
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [inputSize], units: 32, activation: 'relu' }),
            tf.layers.dense({ units: 100, activation: 'softmax' }),
            tf.layers.dense({ units: 20, activation: 'softmax' }),
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
