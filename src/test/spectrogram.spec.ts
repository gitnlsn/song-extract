import { spectrogram } from '../spectrogram';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs';
import { describe, it, expect } from 'vitest';

describe('Spectrogram', () => {
  it('deve gerar um espectrograma a partir de um arquivo WAV', async () => {
    // Preparar: caminho para o arquivo de teste
    const filePath = path.join(__dirname, 'test-assets', 'piano-1-1000.wav');

    // Executar: chamar a função spectrogram com configurações padrão
    const result = spectrogram(filePath, { frameLenght: 300, frameStep: 1000 });

    const magArrays = await result.magnitudes[0].array() as number[][]

    console.log("Result: ", magArrays.length, magArrays[0].length);

    // Verificar: validar o resultado
    expect(result).toBeDefined();
    expect(result.sampleRate).toBeGreaterThan(0);
    expect(result.magnitudes).toBeInstanceOf(Array);
    expect(result.magnitudes.length).toBeGreaterThan(0);

    // Verificar se os magnitudes são tensores do TensorFlow
    result.magnitudes.forEach(magnitude => {
      expect(magnitude instanceof tf.Tensor).toBeTruthy();
      // Verificar se o tensor tem pelo menos 2 dimensões (tempo e frequência)
      expect(magnitude.shape.length).toBeGreaterThanOrEqual(2);
    });
  }, 300000);
});
