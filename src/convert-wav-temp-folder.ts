import { spawn } from 'child_process';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';

export async function convertToWav(inputFilePath: string, props?: { sampleRate?: number }): Promise<string> {
    // Create a temporary directory if it doesn't exist
    const tempDir = path.join(os.tmpdir(), 'song-extract');
    if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir, { recursive: true });
    }

    // Generate output file path
    const outputFileName = `${path.basename(inputFilePath, path.extname(inputFilePath))}.wav`;
    const outputFilePath = path.join(tempDir, outputFileName);

    return new Promise((resolve, reject) => {
        // Spawn ffmpeg process
        const ffmpeg = spawn('ffmpeg', [
            '-i', inputFilePath,
            '-ar', props?.sampleRate?.toString() ?? '1000', // Set sample rate to 16kHz
            '-ac', '1',     // Convert to mono
            '-y',           // Overwrite output file if exists
            outputFilePath
        ]);

        let errorMessage = '';

        ffmpeg.stderr.on('data', (data) => {
            errorMessage += data.toString();
        });

        ffmpeg.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`FFmpeg process exited with code ${code}. Error: ${errorMessage}`));
                return;
            }
            resolve(outputFilePath);
        });

        ffmpeg.on('error', (err) => {
            reject(new Error(`Failed to start FFmpeg process: ${err.message}`));
        });
    });
}
