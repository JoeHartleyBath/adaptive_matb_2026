You should read <a href="https://github.com/juliencegarra/OpenMATB/wiki/How-to-build-a-scenario-file">how to write correctly a scenario file</a> first.

## Presentation of the `communications` plugin.

In the communication task, participants have to pay attention to audio instructions about the frequencies of four radios. The participants regularly receive messages; these may (or may not) be addressed to them. These communications always give precise instructions about the frequency settings of the radios to be monitored.

![The communications task](https://github.com/juliencegarra/OpenMATB/blob/master/.img/communications.png)

## Basic operations

Basically, once the `communications` plugin is started, the participant is auditory prompted via the `radioprompt` parameter, which accepts two values : `own` or `other`. 

An `own` prompt will generate and display a *relevant* auditory prompt, i.e., the participant must react to it. If so, the prompt will mention the callsign of the participant, which is automatically generated at plugin startup, based on the `callsignregex` parameter (`[A-Z][A-Z][A-Z]\d\d\d` by default). An `other` prompt will mention an irrelevant callsign, so the auditory instruction should not be taken into account by the participant.

If you want to force a particular value for the various callsigns (`own` or `other`), you can set it manually via the `owncallsign` and `othercallsign` parameters. Be careful that the `othercallsign` parameter contain the correct number of callsigns, as defined by `othercallsignnumber`.

When prompted, the participant must switch to and tune the correct radio, using left, right, up & down arrow keys. Once done, he or she must hit the `Return` key to confirm the response. The delay left to react can be tuned via the `maxresponsedelay` (ms) parameter.

When relevant (response or timeout), a feedback can be displayed if needed (see the feedback parameters in the table below).

## Automatic solving

To enable/disable a task automation, just use the `automaticsolver` parameter, like in the following:

```
0:00:00;communications;
0:00:00;communications;automaticsolver;True      # The communications is automated as soon as it is started
0:00:30;communications;automaticsolver;False     # Track automation is disabled in the middle of the scenario
0:01:00;communications;stop
```

When enabled, the automation will react to `own` auditory prompt by selecting, tuning and confirming  its response (emulate a `Return` press).

## List of `communications` parameters

|Variable|Description|Possible values|Default|
|:----|:----|:----|:----|
|title|Title of the task, displayed if the plugin is visible|(string)|Communications|
|taskplacement|Task location in a 3x2 canvas|`topleft`, `topmid`, `topright`, `bottomleft`, `bottommid`, `bottomright`, `fullscreen`|bottomleft|
|taskupdatetime|Delay between plugin updates (ms)|(positive integer)|80|
|automaticsolver|When True, the target frequency is automatically set by an automation|(boolean)|False|
|displayautomationstate|If True, the current automation state (MANUAL vs AUTO ON) of the task is displayed.|(boolean)|True|
|callsignregex|Regular expression pattern for callsign generation|(regular expression)|[A-Z][A-Z][A-Z]\d\d\d|
|owncallsign|Callsign of the subject. If empty, automatically generated according to callsignregex|(string)|(empty)|
|othercallsignnumber|Number of irrelevant distracting callsigns|(positive integer)|5|
|othercallsign|List of distracting callsigns. If empty, automatically generated according to callsignregex and othercallsignnumber|(list of string)|(empty)|
|airbandminMhz|Minimum radio frequency, in Mhz|(positive float)|108.0|
|airbandmaxMhz|Maximum radio frequency, in Mhz|(positive float)|137.0|
|airbandminvariationMhz|Minimum frequency variation of the target radio|(positive integer)|5|
|airbandmaxvariationMhz|Maximum frequency variation of the target radio|(positive integer)|6|
|voiceidiom|Voice idiom. The corresponding folder must be present in the Sound directory|`english` or `french`|french|
|voicegender|Voice gender. The corresponding folder must be present in the selected idiom directory|`male` or `female`|female|
|radioprompt|Use it to trigger either a target (own) or a distractor (other) prompt [own or other]. Initially empty. Will be emptied at the end of the prompt|`own` or `other`|(empty)|
|promptlist|List of radio labels, in their order of appearance. Each corresponding file must be available in the sound folder. This list is used both for target and distractors radio lists.|(list of string)|[NAV1, NAV2, COM1, COM2]|
|maxresponsedelay|Maximum response delay (ms)|(positive integer)|20000|
|feedbackduration|Duration (ms) of the feedbacks when enabled|(positive integer)|1500|
|feedbacks-positive-active|Is the positive feedback (correct response) enabled?|(boolean)|False|
|feedbacks-positive-color|Color of the positive feedback if enabled|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|green|
|feedbacks-negative-active|Is the negative feedback (correct response) enabled?|(boolean)|False|
|feedbacks-negative-color|Color of the negative feedback if enabled|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|red|
