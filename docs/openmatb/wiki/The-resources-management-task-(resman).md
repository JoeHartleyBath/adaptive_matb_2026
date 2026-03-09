You should read <a href="https://github.com/juliencegarra/OpenMATB/wiki/How-to-build-a-scenario-file">how to write correctly a scenario file</a> first.

## Presentation of the `resman` plugin.

In this task, participants are faced with a system of pumps and tanks, two of which (referred to as "target tanks") leak. The objective of the participants is to use the different pumps available - by turning them on or off - to correctly compensate for the leaks in the target tanks, and keep them at acceptable levels. To do this, participants must consider the characteristics of the pumps (which vary in flow rate) and the tanks (which vary in capacity). In addition, as the experiment progresses, some pumps may temporarily fail, making the task even more difficult.

Each time a pump is activated (through the corresponding numerical key), its flow (debit per minute) is indicated in the pump status panel, on the right.

![The resources management task](https://github.com/juliencegarra/OpenMATB/blob/master/.img/resman.png)

## Basic operations

In the `resman` task, major experimental manipulations concern `pumps` and `tanks` parameters. For each tank, you can notably modify its:

- initial level (e.g., `tank-c-level` parameter for the tank C)
- maximum capacity (e.g., `tank-c-max`)
- target level if relevant (e.g., `tank-a-target` to modify the target of the tank A)
- depletable nature (e.g., `tank-d-depletable`=False will give you a tank D with infinite capacity)
- leak debit if relevant (e.g., `tank-a-lossperminute`=800)

By default, a `toleranceradius` is set (to 250), which defines the area in which the target tanks should be maintained for a correct performance. If you don‘«÷t want to use a tolerance area, but rather instruct the participant to keep the levels as close as possible of the target level, you can set this `toleranceradius` to 0. If you use the tolerance area, and you want to warn the participant when he or she is outside, you can set a different color to the tolerance area with the `tolerancecoloroutside` parameter.

As for each pump, you can set its:

- flow, i.e., its debit per minute (e.g., `pump-1-flow` to modify flow of the pump n-¶1)
- state, that can be `on`, `off` or `failure` (e.g., `pump-1-state`=`failure` will turn the pump n-¶1 to failure)

You can alter the color associated with each pump state via the `pumpcoloron`, `pumpcoloroff` and `pumpcolorfailure` parameters, respectively.

Note that default values are all acceptable (canonical) and that you should modify these values if you know what you are doing, and how to properly modify the difficulty of the task.

## Automatic solving

To enable/disable a task automation, just use the `automaticsolver` parameter, like in the following:

```
0:00:00;resman;
0:00:00;resman;automaticsolver;True      # The resman is automated as soon as it is started
0:00:30;resman;automaticsolver;False     # Track automation is disabled in the middle of the scenario
0:01:00;resman;stop
```

When enabled, the automation will automatically turn pumps `on` and `off` to compensate leaks in an optimal fashion. Below are the three main rules of this automation:

1. Systematically activate pumps draining non-depletable tanks
2. Activate/deactivate pump whose target tank is too low/high ("Too" means level is out of a tolerance zone around the target level (2500 +/- 150))
3. Balance between the two A/B tanks if sufficient level

## List of `resman` parameters

|Variable|Description|Possible values|Default|
|:----|:----|:----|:----|
|title|Title of the task, displayed if the plugin is visible|(string)|Resources management|
|taskplacement|Task location in a 3x2 canvas|`topleft`, `topmid`, `topright`, `bottomleft`, `bottommid`, `bottomright`, `fullscreen`|bottommid|
|taskupdatetime|Delay between plugin updates (ms)|(positive integer)|2000|
|automaticsolver|When True, the target frequency is automatically set by an automation|(boolean)|False|
|displayautomationstate|If True, the current automation state (MANUAL vs AUTO ON) of the task is displayed.|(boolean)|True|
|toleranceradius|Radius of the tolerance area (0=do not display tolerance area)|(integer)|250|
|tolerancecolor|Color of the tolerance area when volume is inside it|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|black|
|tolerancecoloroutside|Color of the tolerance area when volume is outside it|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|black|
|pumpcoloroff|Color of the pump when its state is `off`|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|white|
|pumpcoloron|Color of the pump when its state is `on`|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|green|
|pumpcolorfailure|Color of the pump when its state is `failure`|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|red|
|displaystatus|Should the pump status be displayed in a specific panel?|(boolean)|True|
|statuslocation|Placement where to display the pump status|`topleft`, `topmid`, `topright`, `bottomleft`, `bottommid`, `bottomright`, `fullscreen`|bottomright|
|tank-a-level|Current level of this tank|(integer)|2500|
|tank-a-max|Maximum level of this tank|(integer)|4000|
|tank-a-target|Target level of this tank|(integer)|2500|
|tank-a-depletable|Is this tank depletable (False=infinite resource)|(boolean)|True|
|tank-a-lossperminute|Volume lost (leak) in a minute|(positive integer)|800|
|tank-b-level|Current level of this tank|(integer)|2500|
|tank-b-max|Maximum level of this tank|(integer)|4000|
|tank-b-target|Target level of this tank|(integer)|2500|
|tank-b-depletable|Is this tank depletable (False=infinite resource)|(boolean)|True|
|tank-b-lossperminute|Volume lost (leak) in a minute|(positive integer)|800|
|tank-c-level|Current level of this tank|(integer)|1000|
|tank-c-max|Maximum level of this tank|(integer)|2000|
|tank-c-target|Target level of this tank|(integer)|(empty)|
|tank-c-depletable|Is this tank depletable (False=infinite resource)|(boolean)|True|
|tank-c-lossperminute|Volume lost (leak) in a minute|(positive integer)|0|
|tank-d-level|Current level of this tank|(integer)|1000|
|tank-d-max|Maximum level of this tank|(integer)|2000|
|tank-d-target|Target level of this tank|(integer)|(empty)|
|tank-d-depletable|Is this tank depletable (False=infinite resource)|(boolean)|True|
|tank-d-lossperminute|Volume lost (leak) in a minute|(positive integer)|0|
|tank-e-level|Current level of this tank|(integer)|3000|
|tank-e-max|Maximum level of this tank|(integer)|4000|
|tank-e-target|Target level of this tank|(integer)|(empty)|
|tank-e-depletable|Is this tank depletable (False=infinite resource)|(boolean)|False|
|tank-e-lossperminute|Volume lost (leak) in a minute|(positive integer)|0|
|tank-f-level|Current level of this tank|(integer)|3000|
|tank-f-max|Maximum level of this tank|(integer)|4000|
|tank-f-target|Target level of this tank|(integer)|(empty)|
|tank-f-depletable|Is this tank depletable (False=infinite resource)|(boolean)|False|
|tank-f-lossperminute|Volume lost (leak) in a minute|(positive integer)|0|
|pump-1-flow|Pump debit per minute|(positive integer)|800|
|pump-1-state|Current state of the pump|`on` or `off` or `failure`|off|
|pump-1-key|Keyboard key to toggle the pump|(keyboard key)|NUM_1|
|pump-2-flow|Pump debit per minute|(positive integer)|600|
|pump-2-state|Current state of the pump|`on` or `off` or `failure`|off|
|pump-2-key|Keyboard key to toggle the pump|(keyboard key)|NUM_2|
|pump-3-flow|Pump debit per minute|(positive integer)|800|
|pump-3-state|Current state of the pump|`on` or `off` or `failure`|off|
|pump-3-key|Keyboard key to toggle the pump|(keyboard key)|NUM_3|
|pump-4-flow|Pump debit per minute|(positive integer)|600|
|pump-4-state|Current state of the pump|`on` or `off` or `failure`|off|
|pump-4-key|Keyboard key to toggle the pump|(keyboard key)|NUM_4|
|pump-5-flow|Pump debit per minute|(positive integer)|600|
|pump-5-state|Current state of the pump|`on` or `off` or `failure`|off|
|pump-5-key|Keyboard key to toggle the pump|(keyboard key)|NUM_5|
|pump-6-flow|Pump debit per minute|(positive integer)|600|
|pump-6-state|Current state of the pump|`on` or `off` or `failure`|off|
|pump-6-key|Keyboard key to toggle the pump|(keyboard key)|NUM_6|
|pump-7-flow|Pump debit per minute|(positive integer)|400|
|pump-7-state|Current state of the pump|`on` or `off` or `failure`|off|
|pump-7-key|Keyboard key to toggle the pump|(keyboard key)|NUM_7|
|pump-8-flow|Pump debit per minute|(positive integer)|400|
|pump-8-state|Current state of the pump|`on` or `off` or `failure`|off|
|pump-8-key|Keyboard key to toggle the pump|(keyboard key)|NUM_8|
