import { convertToWav } from "./convert-wav-temp-folder";

const main = async (props: { inputFilePath: string }) => {
    const wavFilePath = await convertToWav(props.inputFilePath, {sampleRate: 16000});
    console.log(wavFilePath);
}

main({
    inputFilePath: process.argv[2] as string,
}).catch(console.error);