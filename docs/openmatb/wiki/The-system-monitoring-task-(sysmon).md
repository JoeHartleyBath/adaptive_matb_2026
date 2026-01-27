You should read <a href="https://github.com/juliencegarra/OpenMATB/wiki/How-to-build-a-scenario-file">how to write correctly a scenario file</a> first.

## Presentation of the `sysmon`┬áplugin.

In the system monitoring task the participant has to monitor six different gauges (2 lights and 4 scales) that may be subject to failures. It is these failures that the participant is responsible for detecting and recovering through six corresponding response buttons.

![The system monitoring task](https://github.com/juliencegarra/OpenMATB/blob/master/.img/sysmon.png)

## Basic operations
When the system monitoring task is started, you can gauges failures at desired times with the failure parameter of the gauge. For instance, to trigger a failure of the second scale at `0:00:30`, you will use the following command : `0:00:30;sysmon;lights-2-failure;True`.

The failure will be displayed during a delay defined by the `alerttimeout` parameter. In the meantime, subjects are allowed to signal the failure with the corresponding keyboard key, defined by the `lights-2-key` parameter. 

Concerning the scales: in absence of more information, the side of the failure (up or down) will be defined at random. If you want to force a specific location for the incoming failure, use the corresponding `side` parameter. For instance : 

```
0:00:30;sysmon;lights-2-side;up          # The following failure will happen on the top side
0:00:30;sysmon;lights-2-failure;True     # Trigger the failure
```

Either at subject response or at alert timeout, a (positive or negative) feedback might appear during `feedbackduration` milliseconds. Positive and negative feedbacks only concern the scale gauges (no feedback available for lights). Scales feedbacks can be enabled/disabled with either the `feedbacks-positive-active` or `feedbacks-negative-active`. You can also change the color of each (positive or negative) feedback with `feedbacks-positive-color` and ``feedbacks-negative-color`.

You will find more information and features of the plugin below.

## Automatic solving

To enable/disable a task automation, just use the `automaticsolver` parameter, like in the following:

```
0:00:00;sysmon;
0:00:00;sysmon;automaticsolver;True      # The task is automated as soon as it is started
0:00:30;sysmon;automaticsolver;False     # Sysmon automation is disabled in the middle of the scenario
0:01:00;sysmon;stop
```

If enabled, the automation will automatically detect and resolve an ongoing failure, after a delay defined with the `automaticsolverdelay` parameter. 

## List of `sysmon` parameters

|Variable|Description|Possible values|Default|
|:----|:----|:----|:----|
|title|Title of the task, displayed if the plugin is visible|(string)|System monitoring|
|taskplacement|Task location in a 3x2 canvas|`topleft`, `topmid`, `topright`, `bottomleft`, `bottommid`, `bottomright`, `fullscreen`|topleft|
|taskupdatetime|Delay between plugin updates (ms)|(positive integer)|200|
|alerttimeout|Maximum duration of a failure (ms)|(positive integer)|10000|
|automaticsolver|When True, any failure will be automatically corrected after an `automaticsolverdelay` duration|(boolean)|False|
|automaticsolverdelay|Delay (ms) between a failure onset and its automatic correction, if `automaticsolver` == True|(positive integer)|1000|
|displayautomationstate|If True, the current automation state (MANUAL vs AUTO ON) of the task is displayed.|(boolean)|True|
|allowanykey|If True, the subject can use any system monitoring key, to signal a failure. For instance, he or she could use the F2 key to signal a failure for the F4 gauge.|(boolean)|False|
|feedbackduration|Duration (ms) of feedbacks, if enabled|(positive integer)|1500|
|feedbacks-positive-active|Is the positive feedback (correct response) enabled?|(boolean)|True|
|feedbacks-positive-color|Color of the positive feedback if enabled|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|green|
|feedbacks-negative-active|Is the negative feedback (correct response) enabled?|(boolean)|True|
|feedbacks-negative-color|Color of the negative feedback if enabled|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|red|
|lights-1-name|Name of the first (left) light|(string)|F5|
|lights-1-failure|Set this to True if you want to trigger a failure|(boolean)|False|
|lights-1-default|Which is the default state (no failure) of this light?|`on` or `off`|on|
|lights-1-oncolor|Color of the light when it is `on`|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|green|
|lights-1-key|Keyboard key to resolve a failure|(keyboard key)|F5|
|lights-1-on|Current state of the light|(boolean)|True|
|lights-2-name|Name of the second (right) light|(string)|F6|
|lights-2-failure|Set this to True if you want to trigger a failure|(boolean)|False|
|lights-2-default|Which is the default state (no failure) of this light?|`on` or `off`|off|
|lights-2-oncolor|Color of the light when it is `on`|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|red|
|lights-2-key|Keyboard key to resolve a failure|(keyboard key)|F6|
|lights-2-on|Current state of the light|(boolean)|False|
|scales-1-name|Name of the scale #1|(string)|F1|
|scales-1-failure|Set this to True if you want to trigger a failure|(boolean)|False|
|scales-1-side|To which side trigger the failure? (0:random, 1:up, -1:down)|-1, 0, 1|0|
|scales-1-key|Keyboard key to resolve a failure|(keyboard key)|F1|
|scales-2-name|Name of the scale #2|(string)|F2|
|scales-2-failure|Set this to True if you want to trigger a failure|(boolean)|False|
|scales-2-side|To which side trigger the failure? (0:random, 1:up, -1:down)|-1, 0, 1|0|
|scales-2-key|Keyboard key to resolve a failure|(keyboard key)|F2|
|scales-3-name|Name of the scale #3|(string)|F3|
|scales-3-failure|Set this to True if you want to trigger a failure|(boolean)|False|
|scales-3-side|To which side trigger the failure? (0:random, 1:up, -1:down)|-1, 0, 1|0|
|scales-3-key|Keyboard key to resolve a failure|(keyboard key)|F3|
|scales-4-name|Name of the scale #4|(string)|F4|
|scales-4-failure|Set this to True if you want to trigger a failure|(boolean)|False|
|scales-4-side|To which side trigger the failure? (0:random, 1:up, -1:down)|-1, 0, 1|0|
|scales-4-key|Keyboard key to resolve a failure|(keyboard key)|F4|


