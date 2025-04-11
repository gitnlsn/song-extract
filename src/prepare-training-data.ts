import { AudioSegment } from "./interfaces/AudioSegment";
import { Tensor1DInput } from "./interfaces/TensorFlowData";
import { Tensor2DInput } from "./interfaces/TensorFlowData";
import { TrainInput } from "./interfaces/TrainInput";
import { segmentAudio } from "./segmentation";
import * as tf from '@tensorflow/tfjs';

interface PrepareTrainingDataProps {
    trainInputs: TrainInput[];
    inputSize?: number;
}

interface PrepareTrainingDataOutput {
    data: Tensor2DInput;
    labels: Tensor1DInput;
    dataByLabels: Array<{
        label: number;
        data: Tensor2DInput;
    }>
}

/**
 * Shuffles data and labels arrays while maintaining their correspondence
 * @param data Array of data to shuffle
 * @param labels Array of labels to shuffle
 * @param randomizeRounds Number of times to perform the shuffling
 * @returns Object containing shuffled data and labels
 */
function shuffleDataAndLabels<T, U>(
    data: T[], 
    labels: U[], 
    randomizeRounds: number = 1
): { data: T[], labels: U[] } {
    if (data.length !== labels.length) {
        throw new Error('Data and labels arrays must have the same length');
    }
    
    let currentData = [...data];
    let currentLabels = [...labels];
    
    // Perform multiple rounds of shuffling
    for (let round = 0; round < randomizeRounds; round++) {
        const randomizedData: T[] = [];
        const randomizedLabels: U[] = [];
        
        // Create array of indices
        const indices = Array.from({ length: currentData.length }, (_, i) => i);
        
        // Shuffle indices using Fisher-Yates algorithm
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        // Reorder data and labels based on shuffled indices
        indices.forEach(index => {
            randomizedData.push(currentData[index]);
            randomizedLabels.push(currentLabels[index]);
        });
        
        // Update for next round
        currentData = randomizedData;
        currentLabels = randomizedLabels;
    }
    
    return { data: currentData, labels: currentLabels };
}

export const prepareTrainingData = async ({
    trainInputs,
    inputSize
}: PrepareTrainingDataProps) => {
    // Prepare data and labels
    const data: number[][] = [];
    const labels: number[] = [];
    const dataByLabels: Array<{
        label: number;
        data: number[][];
    }> = [];

    for (const trainInput of trainInputs) {
        const segments = await segmentAudio(trainInput.filePath, {
            sampleRate: inputSize ?? 1000,
            segmentDuration: 1
        });

        segments.forEach(segment => {
            // Create a new array with the maximum length
            const paddedData = new Float32Array(inputSize ?? 1000);
            // Copy the original data
            paddedData.set(segment.decoded);
            // Fill the rest with zeros if needed
            data.push(Array.from(paddedData));
            
            // Set label based on whether the segment is within song intervals
            labels.push(trainInput.type);

            // Add the data to the dataByLabels array
            const existingByLabelsItem = dataByLabels.find(item => item.label === trainInput.type);
            if (existingByLabelsItem) {
                existingByLabelsItem.data.push(Array.from(paddedData));
            } else {
                dataByLabels.push({
                    label: trainInput.type,
                    data: [Array.from(paddedData)]
                });
            }
        });
    }
    
    // Randomize data and labels 100 times while maintaining correspondence
    const { data: randomizedData, labels: randomizedLabels } = shuffleDataAndLabels(data, labels, 100);
    
    return {
        data: randomizedData,
        labels: randomizedLabels,
        dataByLabels
    } satisfies PrepareTrainingDataOutput;
};
