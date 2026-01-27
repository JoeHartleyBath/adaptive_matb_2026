You should read <a href="https://github.com/juliencegarra/OpenMATB/wiki/How-to-build-a-scenario-file">how to write correctly a scenario file</a> first.

## Presentation of the `scheduling` plugin.

The scheduling plugin is not a task itself but a time display that participants can use to visualize certain information such as the start and stop of tasks, or the start and stop of automations.

![The scheduling panel](https://github.com/juliencegarra/OpenMATB/blob/master/.img/scheduling.png)

## Basic operations

Mainly, you can modify the duration displayed, by modifying the `minduration` parameter (8 minutes by default).

The `scheduling` plugin will automatically display the timeline of the started task, but you can decide to change this list by modifying the `displayedplugins` parameter (list of task alias).

You can also decide to hide the chronometer by setting `displaychronometer`=False. Finally, this chronometer can be reversed to display the remaining time, with the `reversechronometer` parameter.

## List of `scheduling` parameters

|Variable|Description|Possible values|Default|
|:----|:----|:----|:----|
|title|Title of the task, displayed if the plugin is visible|(string)|Scheduling|
|taskplacement|Task location in a 3x2 canvas|`topleft`, `topmid`, `topright`, `bottomleft`, `bottommid`, `bottomright`, `fullscreen`|topright|
|taskupdatetime|Delay between plugin updates (ms)|(positive integer)|1000|
|minduration|Duration (minutes) of the displayed scheduling|(positive integer)|8|
|displaychronometer|Should the elapsed time be displayed?|(boolean)|True|
|reversechronometer|Should elapsed time be turned to remaining time?|(boolean)|False|
|displayedplugins|List of plugin schedules to display|(list of task plugins alias = `sysmon`, `track`, `communications`, `resman`)|[`sysmon`, `track`, `communications`, `resman`]|
