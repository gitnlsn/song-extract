import { describe, it, expect } from 'vitest';
import { convertToWav } from '../convert-wav-temp-folder';
import { segmentAudio } from '../segmentation';
import { trainModel } from '../train-model';
import path from 'path';
import { prepareTrainingData } from '../prepare-training-data';
import { TrainInput } from '../interfaces/TrainInput';
import * as tf from '@tensorflow/tfjs';

describe('Train Model', () => {
    it('should train the model', async () => {
        const sampleRate = 1000;

        const tSet1: TrainInput = {
            filePath: path.join(__dirname, 'test-assets', 'piano-1-1000.wav'),
            type: 1
        }
        // const tSet2: TrainInput = {
        //     filePath: path.join(__dirname, 'test-assets', 'piano-2-4000.wav'),
        //     type: 'song'
        // }
        const tSet3: TrainInput = {
            filePath: path.join(__dirname, 'test-assets', 'speech-1-1000.wav'),
            type: 0
        }
        // const tSet4: TrainInput = {
        //     filePath: path.join(__dirname, 'test-assets', 'speech-2-4000.wav'),
        //     type: 'speech'
        // }

        const { data, labels, dataByLabels } = await prepareTrainingData({ trainInputs: [tSet1, tSet3], inputSize: sampleRate });

        console.log("Started training model with: ", data.length, " samples size");

        const model = await trainModel({ data, labels, props: { epochs: 5, batchSize: 8192, inputSize: sampleRate } });

        for (const dataByLabel of dataByLabels) {
            const trialData = dataByLabel.data.slice(0, 3);
            const expectedLabel = dataByLabel.label;

            const trialDataTensor = tf.tensor2d(trialData);

            const prediction = model.predict(trialDataTensor);
            console.log("Prediction result: ", expectedLabel, prediction.toString());
        }

        expect(model).toBeDefined();
    }, 300000);
});