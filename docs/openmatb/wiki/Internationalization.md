One of the most critical aspects of portability is translation. OpenMATB users should be able to use the software in their native language or, if necessary, easily create the translated version.

### Table of contents
- [Translate visuals](#how-to-translate-visuals-)
- [Translate audio files](#how-to-translate-the-audio-files-)

## How to translate visuals ?
To translate the visual interface of the OpenMATB, one should set the variable `language` in the `config.txt` file. For instance `language=fr_FR` will enable french. The argument corresponds to the name of the translation file, in the `Translations` folder.

Therefore, if you wish to implement a new language : 
1. Copy-paste an existing translation file
2. Rename it, using the correct language code (e.g., `fr_FR.txt`, see [here](http://quivi.sourceforge.net/languagecodes.html) for more information)
3. Fill each line of the new file with the new target language
4. The file must be located in the `Translations` folder
5. Don't forget to specify the language you want to use in the `config.txt` file. The argument should correspond to the name of the file you have just created

## How to translate the audio files ?

### How to call for a specific audio file ?
The audio files are located in the `Sounds` folder, according to their idiom and their gender (male, female). By defaults, the voice idiom is `english` and its gender is `male`. If one wishes to select another voice, one should state it explicitly in the scenario. For instance, if a french female voice is required, one should add the following two lines in the scenario : 

```
0:00:00;communications;voicegender;female
0:00:00;communications;voiceidiom;french
```
Note that the arguments (`female`, `french`) should be exactly the same as the names of the folders that contain the required files.

### Build a new voice
In OpenMATB, every communication audio file is built on the fly, using a combination of elementary audio files (letters, digits, instructions...), with the following general syntax : "[CALLSIGN], [CALLSIGN],  turn your [RADIO] radio to frequency [FREQUENCY]". For instance : "A-B-D-1, A-B-D-1, turn your COM-1 radio to frequency 1-2-0 point 3"

Hence, the only thing to do when an original voice is required is to record the following audio samples and put them in the corresponding folder : 
* Letters : A, B, C, ..., X, Y, Z (a.wav, b.wav, c.wav, ..., x.wav, y.wav, z.wav)
* Digits : 0, 1, 2, ..., 7, 8, 9 (0.wav, 1.wav, 2.wav, ..., 7.wav, 8.wav, 9.wav)
* Radio names : COM 1, COM 2, NAV 1, NAV 2 (com1.wav, com2.wav, nav1.wav, nav2.wav)
* The expressions : "turn your", "radio to frequency" and "point" (radio.wav, frequency.wav)

Obviously, depending on the language, you may have to adapt the linking expressions (e.g., "turn your"). Moreover, it could be that in your mother tongue, the basic syntax of such a phrase is really different. In that case, do not hesitate to contact us, so we can implement the relevant syntax in our audio file generator.

### The recording
For the record, we recommend to use a free audio recording software like [Audacity](https://audacity.fr/), and to record the entire list at one time. Do not hesitate to leave some silence between two sounds, so we can easily distinguish sounds and split your recording into multiple small files.

![OpenMATB : record new audio files with Audacity](https://user-images.githubusercontent.com/10955668/49248292-ab4b5200-f419-11e8-8cdc-ba8bd0932f23.png)

For the ease of further processing (see below), please note the recording sequence or follow the default sequence if possible, so we can process the record correctly, even if we do not understand the language.

The default recording sequence is the following (adapt it to your idiom !) : 
(each comma is a silence)

"Turn your, radio to frequency, com 1, com 2, nav 1, nav 2, point, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z"

Be careful to keep the intonation equal throughout the recording and not to pronounce the end of the list as the end of a sentence. Indeed, each recorded piece can be used at any point of the generated sentence.


### Audio files properties
Each audio file should be a **mono**, **32-bits wav** file, sampled at **44100 Hz**.

### Automatic splitting
We developed a bit of code to handle automatic splitting of our records. You can adapt it to your own use if you have sufficient skills with python or, alternatively, you can send us your record and we'll split it for you.

The use of this script is mandatory for reason of sounds consistency (e.g., the script add brown noise to every piece of audio).

The python script is based on a useful linux library called [SoX](https://linux.die.net/man/1/sox).
Among the many features of SoX, there is the possibility to analyze an audio record, locate silences and use them to split the record into multiple audio files.

```python
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import os,sys
from subprocess import call
import pdb

target_list = ['radio','frequency','com_1','com_2','nav_1','nav_2','point', '0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h', 'i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

gender = 'male'
langage = 'french'
input_name = 'benoit.wav'

preproc_commands = [['highpass', '200'],['lowpass', '1500']]
options = []

for this_command in preproc_commands:
    options.append(this_command[0])
    output_name = input_name + '_' + '_'.join(options) + '.wav'
    call(['sox', input_name, output_name] + this_command)
    input_name = output_name

# Splitting audio
options.append('cut')
input_name = output_name
output_name = audio_name + '_' + '_'.join(options) + '.wav'
call(['sox', input_name, output_name, 'silence', '1', '0.01', '0.2%', '1', '0.1', '0.2%', ':', 'newfile', ':', 'restart'])

# Move all resulting file in a specific folder
current_path = os.path.dirname(os.path.realpath(__file__))
target_path = current_path + os.sep + folder
list_of_files = [files for files in os.listdir(current_path) if options[-1] in files]
if not os.path.exists(current_path + os.sep + folder):
    os.makedirs(current_path + os.sep + folder)
else: # Empty it
    for this_file in os.listdir(target_path):
        os.remove(target_path + os.sep + this_file)
for this_file in list_of_files:
    file_path = current_path + os.sep + this_file
    this_size = os.path.getsize(file_path)
    # Remove files that are too short
    if this_size < 20000:
        os.remove(file_path)
    else:
        os.rename(file_path, target_path + os.sep + this_file)
    
# Break if not correct number of audio files, and remove them
list_of_files = [files for files in os.listdir(target_path)]
if len(list_of_files) != 43:
    print "Not 43 audio files... Try again"
    for this_file in list_of_files:
        os.remove(target_path + os.sep + this_file)

# Adding brown noise to each audio file
for f,this_file in enumerate(sorted(list_of_files)):
    input_path = target_path + os.sep + this_file
    output_path = target_path + os.sep + target_list[file_dict[audio_name]['order']][f] + '.wav'
    call(['sox', input_path, 'this_noise.wav', 'synth', 'brownnoise', 'vol','0.08'])
    call(['sox','-m','this_noise.wav', input_path, output_path])
    os.remove(input_path)
    os.remove('this_noise.wav')

# Finally, copy a good noise.wav version into the working directory
call(['cp', 'brownnoise500ms.wav', target_path + os.sep + 'empty.wav'])
```

