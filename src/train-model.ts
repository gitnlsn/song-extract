import * as tf from '@tensorflow/tfjs';
import { AudioSegment } from './interfaces/AudioSegment';
import { TrainInput } from './interfaces/TrainInput';

function isInSongInterval(segment: AudioSegment, intervals: TrainInput['songIntervals']): boolean {
    return intervals.some(interval => 
        segment.startTime >= interval.start && segment.endTime <= interval.end
    );
}

export async function trainModel(segments: AudioSegment[], trainInput: TrainInput): Promise<tf.LayersModel> {
    // Prepare data and labels
    const data: number[][] = [];
    const labels: number[] = [];

    segments.forEach(segment => {
        // Convert Float32Array to regular array (values are already normalized between -1 and 1)
        data.push(Array.from(segment.decoded));
        
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
            tf.layers.dense({inputShape: [1000], units: 32, activation: 'relu'}),
            tf.layers.dense({units: 2, activation: 'softmax'}), // Changed to 2 units for binary classification
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
        epochs: 5,
        batchSize: 32,
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
