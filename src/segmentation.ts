import * as fs from 'fs';
import * as wav from 'node-wav';
import { AudioSegment } from './interfaces/AudioSegment';

export async function segmentAudio(filePath: string, props?: { sampleRate?: number, segmentDuration?: number }): Promise<AudioSegment[]> {
    // Read the WAV file
    const buffer = fs.readFileSync(filePath);
    const result = wav.decode(buffer);
    
    const sampleRate = props?.sampleRate ?? 1000;
    const segmentDuration = props?.segmentDuration ?? 1;

    if (!result) {
        throw new Error('Failed to decode WAV file');
    }

    const { channelData, sampleRate: fileSampleRate } = result;
    
    // Verify if the provided sample rate matches the file's sample rate
    if (fileSampleRate !== sampleRate) {
        throw new Error(`Sample rate mismatch. Expected ${sampleRate}, got ${fileSampleRate}`);
    }

    const segments: AudioSegment[] = [];
    const samplesPerSegment = sampleRate * segmentDuration;
    const overlapSamples = Math.floor(samplesPerSegment / 2); // 50% overlap
    const stepSize = samplesPerSegment - overlapSamples; // How much to move forward each iteration
    const totalSamples = channelData[0].length;
    
    // Process segments with overlap
    for (let startSample = 0; startSample < totalSamples; startSample += stepSize) {
        const endSample = Math.min(startSample + samplesPerSegment, totalSamples);
        const startTime = startSample / sampleRate;
        const endTime = endSample / sampleRate;
        
        // Create a new Float32Array for this segment
        const segmentData = new Float32Array(endSample - startSample);
        
        // Copy the samples for this segment
        for (let i = startSample; i < endSample; i++) {
            segmentData[i - startSample] = channelData[0][i];
        }
        
        segments.push({
            startTime,
            endTime,
            decoded: segmentData
        });

        // If we've reached the end of the file, break
        if (endSample >= totalSamples) {
            break;
        }
    }

    return segments;
}
