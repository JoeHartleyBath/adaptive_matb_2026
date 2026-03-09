You should read <a href="https://github.com/juliencegarra/OpenMATB/wiki/How-to-build-a-scenario-file">how to write correctly a scenario file</a> first.

## Presentation of the `track` plugin.

In the tracking task, the participant must use a joystick to compensate the drift of a cursor, so as to keep it in a defined (tolerance) squared area, whose size can vary.

![The tracking task](https://github.com/juliencegarra/OpenMATB/blob/master/.img/track.png)

## Basic operations

When the tracking task is started, the cursor moves following a drifting function which defines a pseudo-random path, i.e., the movement of the cursor seems random to the participant, but it is deterministic in absence of any participant (joystick) input.

The proportion of the target (or tolerance) area can defined via the `targetproportion` parameter (see below). When outside of this area, the cursor color changes to `cursorcoloroutside` parameter value. If you wish the color to remain the same, you should just make `cursorcolor` and `cursorcoloroutside` values equal.

By default, the joystick actions are not "reversed", i.e., pushing the joystick causes the cursor to move up. If you want to get closer to the controls of an aircraft, i.e., the cursor moves down if you push the joystick, then you can simply set `inverseaxis` to `True`.

## Automatic solving

To enable/disable a task automation, just use the `automaticsolver` parameter, like in the following:

```
0:00:00;track;
0:00:00;track;automaticsolver;True      # The task is automated as soon as it is started
0:00:30;track;automaticsolver;False     # Track automation is disabled in the middle of the scenario
0:01:00;track;stop
```

If enabled, the automation will automatically make the cursor sticking to the center of the target.

## List of `track` parameters

|Variable|Description|Possible values|Default|
|:----|:----|:----|:----|
|title|Title of the task, displayed if the plugin is visible|(string)|Tracking|
|taskplacement|Task location in a 3x2 canvas|`topleft`, `topmid`, `topright`, `bottomleft`, `bottommid`, `bottomright`, `fullscreen`|topmid|
|taskupdatetime|Delay between plugin updates (ms)|(positive integer)|20|
|cursorcolor|Color of the moving cursor|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|black|
|cursorcoloroutside|Color of the moving cursor when outside the target area|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|red|
|automaticsolver|When True, the cursor movement is automatically compensated toward the center|(boolean)|False|
|displayautomationstate|If True, the current automation state (MANUAL vs AUTO ON) of the task is displayed.|(boolean)|True|
|targetproportion|Radius proportion of the target area. 0.1 means that the radius of the target area is 10% of the task total width. 0 means no target area at all.|(unit_interval=[0:1])|0.25|
|joystickforce|The smaller this factor, the more the joystick movement is attenuated. Greater values leads to a more sensitive joystick.|(integer)|1|
|inverseaxis|Set this to True if joystick actions should be inverted|(boolean)|False|
