export interface AudioSegment {
    startTime: number;
    endTime: number;
    decoded: Float32Array;
}