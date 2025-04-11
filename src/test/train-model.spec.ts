import { describe, it, expect } from 'vitest';
import { sampleAudioTrainInput } from './test-assets/sample-audio-train-input';
import { convertToWav } from '../convert-wav-temp-folder';
import { segmentAudio } from '../segmentation';
import { trainModel } from '../train-model';
import path from 'path';

describe('Train Model', () => {
    it('should train the model', async () => {

        const filePath = path.join(__dirname, 'test-assets', 'sample-audio.wav');
        const segments = await segmentAudio(filePath);

        console.log(segments[0].decoded.length);

        const model = await trainModel(segments, sampleAudioTrainInput, { epochs: 1, batchSize: 512 });

        console.log(model);

        expect(model).toBeDefined();
    }, 120000);
});