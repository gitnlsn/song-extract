export interface TrainInput {
    filePath: string

    songIntervals: {
        start: number;
        end: number;
    }[];
}