You should read <a href="https://github.com/juliencegarra/OpenMATB/wiki/How-to-build-a-scenario-file">how to write correctly a scenario file</a> first.

## Presentation of the `performance` plugin.

<img align="right" src="https://github.com/juliencegarra/OpenMATB/blob/master/.img/performance.png">

Often, it is necessary in MATB to maintain participant engagement into the task, by feed them back, with their current performance level. The `performance` intends to do it, via a simple bar.

## Basic operations

For now, the functioning of the performance plugin is the following: a performance level is computed for each task (see below), and, as MATB intends to test multitask abilities, it displays the worse performance computed on all the task at hand. The plugin allows to display a critical level, with the `criticallevel` parameter. When the performance goes below this threshold, the bar color will turn to `criticalcolor`. By default, performance level is blocked when it passes below the critical level, but you can decide to display it anyway with the `shadowundercritical` parameter set to False.

## Perfomance computation

For the global performance to be displayed, each task performance must be computed and normalized. Note that the following decisions are arbitrary and based on our expertise with the MATB. If you want to modify the way the various performance scores are computed, please refer to [this page](https://github.com/juliencegarra/OpenMATB/wiki/How-to-modify-automatic-performance-computation%3F).

`sysmon` ÔÇô Only considering hits and misses for system monitoring. Compute average of 4 last signal detection events and map it to a 0:1 scale. Hitting the four last events will lead to maximum performance. In consequence, no performance score is available until four `sysmon` events have been triggered.

`track` ÔÇô Time proportion spent in target for the last 5 seconds. No performance is available until 5 seconds of tracking task have elapsed.

`resman` ÔÇô Time proportion spent in target for the last 5 seconds. No performance is available until 5 seconds of tracking task have elapsed.

`communications` ÔÇô Each response can be considered either correct or incorrect. So communications score is here considered as the correct proportion of the four last responses. In consequence, no performance score is available until four `communications` (own) events have been triggered.

**NB:** More parameters will be proposed in the next versions to allow the modification of either the number of events or the duration considered in the performance computation.

## List of `performance` parameters
|Variable|Description|Possible values|Default|
|:----|:----|:----|:----|
|title|Title of the task, displayed if the plugin is visible|(string)|Performance|
|taskplacement|Task location in a 3x2 canvas|`topleft`, `topmid`, `topright`, `bottomleft`, `bottommid`, `bottomright`, `fullscreen`|topright|
|taskupdatetime|Delay between plugin updates (ms)|(positive integer)|50|
|levelmin|Minimum performance level|(integer)|0|
|levelmax|Maximum performance level|(integer)|100|
|ticknumber|Steps (visual) from levelmin to levelmax|(integer)|5|
|criticallevel|Level below which performance is considered critical|(integer)|20|
|shadowundercritical|Should the performance level be shadowed when under criticallevel?|(boolean)|True|
|defaultcolor|Color of the fluctuating performance bar|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|green|
|criticalcolor|Color of the performance bar when performance is critical|`white`, `black`, `green`, `red`, `background`, `lightgrey`, `grey`, `blue`|red|

