import { Tensor1DInput } from "./interfaces/TensorFlowData";
import { Tensor2DInput } from "./interfaces/TensorFlowData";
import { TrainInput } from "./interfaces/TrainInput";
import { spectrogram } from "./spectrogram";

interface PrepareTrainingDataProps {
    trainInputs: TrainInput[];
    frameLenght?: number;
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
    frameLenght
}: PrepareTrainingDataProps) => {
    const frameLenghtResolved = frameLenght ?? 1000;

    // Prepare data and labels
    const data: number[][] = [];
    const labels: number[] = [];
    const dataByLabels: Array<{
        label: number;
        data: number[][];
    }> = [];

    for (const trainInput of trainInputs) {
        const { magnitudes } = spectrogram(trainInput.filePath, {
            frameLenght: frameLenghtResolved,
            frameStep: Math.max(frameLenghtResolved / 100, 100)
        });

        for (const magnitude of magnitudes) {
            const magnitudes1D = await magnitude.array() as number[][]

            data.push(...magnitudes1D);
            labels.push(...Array(magnitudes1D.length).fill(trainInput.type));

            const existingByLabelsItem = dataByLabels.find(item => item.label === trainInput.type);
            if (existingByLabelsItem) {
                existingByLabelsItem.data.push(...magnitudes1D);
            } else {
                dataByLabels.push({
                    label: trainInput.type,
                    data: magnitudes1D
                });
            }
        }
    }

    // Randomize data and labels 100 times while maintaining correspondence
    const { data: randomizedData, labels: randomizedLabels } = shuffleDataAndLabels(data, labels, 100);

    return {
        data: randomizedData,
        labels: randomizedLabels,
        dataByLabels,
    } satisfies PrepareTrainingDataOutput;
};
