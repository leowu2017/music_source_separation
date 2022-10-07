import pathlib
import time
import torch
import librosa
import soundfile as sf
import typing

from bytesep.models.lightning_modules import get_model_class
from bytesep.separator import Separator

SAMPLE_RATE = 44100


def user_defined_build_separator() -> Separator:
    r"""Users could modify this file to load different models.

    Returns:
        separator: Separator
    """

    input_channels = 1
    output_channels = 1
    target_sources_num = 1
    segment_samples = int(44100 * 30.)
    batch_size = 1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_type = "ResUNet143_Subbandtime"

    if model_type == "ResUNet143_Subbandtime":
        checkpoint_path = pathlib.Path(pathlib.Path.home(), "git/music_source_separation/checkpoints/musdb18/train/config=vocals-accompaniment,resunet_subbandtime,gpus=1",
                                       "step=170000.pth")

    elif model_type == "MobileNet_Subbandtime":
        checkpoint_path = pathlib.Path(pathlib.Path.home(), "git/music_source_separation/checkpoints/musdb18/train/config=vocals-accompaniment,resunet_subbandtime,gpus=1",
                                       "step=170000.pth")

    # Get model class.
    Model = get_model_class(model_type)

    # Create model.
    model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        target_sources_num=target_sources_num,
    )

    # Load checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])

    # Move model to device.
    model.to(device)

    # Create separator.
    separator = Separator(
        model=model,
        segment_samples=segment_samples,
        batch_size=batch_size,
        device=device,
    )

    return separator


def infer(inputPath: str, outputPath: str, separator: Separator) -> typing.NoReturn:
    # audio
    audio, _ = librosa.load(inputPath, mono=True, sr=44100)
    audio = audio[None, ...]
    input_dict = {'waveform': audio}

    # Separate
    sep_audio = separator.separate(input_dict)
    sep_audio = sep_audio[0, :]
    sf.write(outputPath, sep_audio, SAMPLE_RATE)


def main():
    r"""An example of using bytesep in your programme. After installing bytesep, 
    users could copy and execute this file in any directory.
    """

    inputDir = pathlib.Path(pathlib.Path.home(), "DL/real")
    outputDir = pathlib.Path(pathlib.Path.home(), 'DL/pred')

    # Build separator.
    separator = user_defined_build_separator()

    # get all inputs
    inputs = inputDir.rglob('*.wav')

    # generate output folder
    outputDir.mkdir(parents=True, exist_ok=True)

    separate_time = time.time()
    # iterate over inputs
    for inputPath in inputs:
        print(f"Prcessing '{inputPath}'")
        outputName = inputPath.with_stem(inputPath.stem + '_se').name
        outputPath = pathlib.Path(outputDir, outputName)

        infer(inputPath, outputPath, separator)

    # done
    print("Done! {:.3f} s".format(time.time() - separate_time))


if __name__ == "__main__":

    main()
