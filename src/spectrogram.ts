import * as fs from 'fs';
import * as wav from 'node-wav';
import * as tf from '@tensorflow/tfjs';

interface Spectrogram {
    magnitudes: tf.Tensor<tf.Rank>[];
    sampleRate: number;
}

export function spectrogram(filePath: string, props?: { frameLenght?: number, frameStep?: number }): Spectrogram {
    // Read the WAV file
    const buffer = fs.readFileSync(filePath);
    const result = wav.decode(buffer);

    if (!result) {
        throw new Error('Failed to decode WAV file');
    }

    const { channelData, sampleRate } = result;

    const magnitudes = channelData.map(data => {
        const audioTensor = tf.tensor1d(data);

        const spectrogram = tf.signal.stft(
            audioTensor,
            props?.frameLenght ?? 1024,
            props?.frameStep ?? 512,
            props?.frameLenght ?? 1024
        );

        return tf.abs(spectrogram);
    })

    console.log("Magnitudes: ", magnitudes[0].shape);

    return { magnitudes, sampleRate };
}
