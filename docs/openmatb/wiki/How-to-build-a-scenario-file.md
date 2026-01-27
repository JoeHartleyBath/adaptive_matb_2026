### Table of contents
- [Scenario syntax](#scenario-syntax)
 * [Time](#time)
 * [Object](#object)
 * [Command](#command)
- [Recommended structure of a scenario file](#recommended-structure-of-a-scenario-file)
- [Example](#example)

## Scenario syntax

Scenario files must be placed into the `includes/scenarios/` directory. The relative path of the scenario to execute at OpenMATB startup, must be indicated into the `config.ini` file (`scenario_path` parameter). A scenario can be placed into a subfolder. For instance, `scenario_path = myexperiment/scenario_1.txt`

When OpenMATB is launched, the `scenario_path` is browsed and launched. A session ID is displayed, as well as potential errors (listed in `last_scenario_errors.log`). From there, each and every experimental event will be handled through this *scenario file*, which contains a list of *commands*.

For instance, consider the following piece of scenario :

```
0:00:00;sysmon;start
0:00:10;sysmon;scale-1-side;1
0:00:10;sysmon;scale-1-failure;True
0:00:30;sysmon;light-2-failure;True
0:00:50;sysmon;stop
```

These lines correspond respectively to the following actions:

1. Start the system monitoring task at OpenMATB startup (0:00:00)
2. Trigger the failure (to the top) of the first scale at the 10th second
3. Trigger the failure of the second light at the 30th second
4. Stop the system monitoring task at the 50th second

Aside from the lines that are ignored (empty or beginning with a `#`), each scenario line is composed by at least three instructions that are separated by a semicolon : `time;plugins;command`.


### Time

Time comes first and must be formatted as `H:mm:ss`. It corresponds to when the command must be triggered, relative to the start of the experiment.

### Plugins alias

The plugins part states which plugin is concerned. Each plugin is associated with an alias, which must be used into the scenario file. Some plugins are visible (tasks) and other are invisible (e.g., the `parallelport` plugin). Below is the list of the available objets, their alias, and what their are.

 * System monitoring (`sysmon`): a task where the subject has to monitor six gauges and detect their potential malfunctions;
 * Tracking (`track`): a task where the subject must use a joystick to compensate for the drift of a moving reticulum;
 * Scheduling view (`scheduling`): a display of when the tasks are handled manually or automatically;
 * Communication (`communications`): an auditory task where the subject must attend to message indicating how to tune four radio frequencies;
 * Resources management (`resman`): a task where the subject must manage a tank/pump system, to compensate for the leak of two target tanks;
 * Performance (`performance`): a display of what is the current general performance level of the subject;
 * Instructions (`instructions`): a fullscreen plugin that is able to display various text instructions from text files (located in the `includes/instructions` directory);
 * Generic Scales (`genericscales`): a module that can display a custom questionnaire from text files (located in the `includes/questionnaires` directory);
 * Lab streaming layer (`labstreaminglayer`): a module that allows the user to stream markers, as well as the whole log information, accross the network with the LSL protocol;
 * Parallel port (`parallelport`): a module that allows the user to send markers through a physical parallel port.

### Command

Once the time and the plugins have been informed, the user must specify the command to be sent. There are two types of commands : *actions* and *parameters*.

#### Actions

Actions are commands that intend to send a specific signal to a plugin. For instance, if one wants to send a *pause* signal to the resources management task, at a given point in time (e.g., `0:05:30`) he/she must type the following line:


```
0:05:30;resman;pause
```

Possible actions are limited in number. In general, the most used actions are `start` and `stop`. If one omits to *start* a given task, it simply won't be displayed. In the same way, each started task must be stopped.

*Note 1:* Some plugins are automatically *stopped* once their are completed by the subject (`instructions`, `genericscales`).

*Note 2:* When a plugin is sent an action instruction (e.g., `start`), what happens is that the software execute the corresponding method in the plugin python file. For instance, the above scenario line leads the software to execute the `pause()` method of the `resman.py` file.

#### Parameters

Parameters are plugin states that are alterable. These are describing their internal functioning and can be altered at any time, even if the plugin has not been started yet. Consider the following example:

```
0:00:10;sysmon;scales-1-failure;True
```
Here, the user decided to trigger the failure of the scale 1, in the system monitoring task, at the 10th second. You should have noticed that the command is splitted into two parts (`scales-1-failure` and `True`). The first indicates the parameter to be modified, the second is the value the parameter is given. Depending on the plugin, some parameters must be accessed directly or through a hierarchical structure. For instance `scales-1-failure` stipulates that we want to modify the `failure` parameter of the `1`st element of the various `scales`.

This corresponds to the way parameters are organized in each task plugin. For instance, the command we just mentionned correspond to the way information is organized in the `sysmon.py` file:

```python
[...]
   alerttimeout=10000,
   lights=dict([('1', dict(name='F5', failure=False, default='on',
                 oncolor=C['GREEN'], key='F5', on=True)),
                ('2', dict(name='F6', failure=False, default='off',
                 oncolor=C['RED'], key='F6', on=False))]),

   scales=dict([('1', dict(name='F1', failure=False, side=0, key='F1')),
                ('2', dict(name='F2', failure=False, side=0, key='F2')),
                ('3', dict(name='F3', failure=False, side=0, key='F3')),
                ('4', dict(name='F4', failure=False, side=0, key='F4'))])
[...]
```

Here, you can see that if we want to change the value of `failure` for the first scale, we have to go through a hierarchy in the right order, that is, first `scales`, then `1`, and finally `failure`. The various components of a hierarchy structure must be separated by hyphen, hence `scales-1-failure`.

On the other hand, we can see that some parameters do not pertain to any hierarchy structure (e.g., `alerttimeout`), so we can alter them more directly. For instance:

```
0:00:10;sysmon;alerttimeout;2000
```

Thus, to facilitate the use of the scenario file, one should know the different parameters available for each task, as well as the values that these parameters accept. A complete dictionary of available parameters is available [here](#parameter-dictionary).


## Recommended structure of a scenario file
When one executes a scenario file in OpenMATB, all the commands are first ordered as a function of their time of execution. As a consequence, one could overlook the order in which the scenario file is written. However, for the sake of clarity, and given the important number of commands such a scenario file could contain, we strongly recommend the organization mentioned below. This scenario example is empty. It only contains comments (#) that are structuring it. A more fully example is available below.

```
# Scenario title
# Comments

# 1. Set tasks parameters
# 1.a. System monitoring parameters
# 1.b. Resources management parameters
# 1.c. Tracking parameters
# 1.d. Communications parameters

# 2. Start tasks that will be used

# 3. Set scenario events

# 3.a. System monitoring events
# 3.b. Resources management events
# 3.c. Tracking events
# 3.d. Communications events

# 4. Stop tasks

```

If you want to start a task at, say, 2 minutes, itÔÇÖs important to start it at OpenMATB startup anyway, like this :

```
# Scenario title
# Comments

[ÔÇª]

# 2. Start tasks that will be used
# Say you want to start the sysmon task only at 2 minutes
# Just start, pause and hide itÔÇª
0:00:00;sysmon;start
0:00:00;resman;start
0:00:00;sysmon;pause
0:00:00;sysmon;hide

# 3. Set scenario events

# 3.a. System monitoring events
# Recover the sysmon task at 0:02:00
0:02:00:sysmon;resume
0:02:00:sysmon;show

# 3.b. Resources management events
[ÔÇª]

# 4. Stop tasks
0:00:00;sysmon;stop
0:00:00;resman;stop
```

## Example (1-minute scenario)

```
# Scenario template
# Durations are expressed in milliseconds
# Timestamps are stated in the following format H:MM:SS

# 1. Set tasks parameters

# 1.a. System monitoring parameters
# Unable sysmon feedbacks
0:00:00;sysmon;feedbacks-positive-active;False
0:00:00;sysmon;feedbacks-negative-active;False

# 1.b.i. Resources management parameters
# Change tank B target in resman
0:00:00;resman;tank-b-target;1000


# 1.c. Tracking parameters
# Change tracking cursor color
0:00:00;track;cursorcolor;#009900

# 1.d. Communications parameters
# Change the callsign format, the number of irrelevant call-signs, the voice gender and idiom
0:00:00;communications;callsignregex;[A-Z][A-Z]\d\d
0:00:00;communications;othercallsignnumber;5
0:00:00;communications;voicegender;male
0:00:00;communications;voiceidiom;french


# 2. Start appropriate tasks
0:00:00;resman;start
0:00:00;track;start
0:00:00;sysmon;start
0:00:00;scheduling;start
0:00:00;communications;start


# 3. Set scenario events
# Display a NASA-TLX questionnaire

0:00:00;genericscales;filename;nasatlx_fr.txt
0:00:00;genericscales;start


# 3.a. System monitoring events
# Schedule some gauge failures
0:00:10;sysmon;scales-1-side;-1
0:00:10;sysmon;scales-1-failure;True
0:00:30;sysmon;scales-4-side;1
0:00:30;sysmon;scales-4-failure;True

# 3.b. Resources management events
# Schedule automatic solving and pump failures
0:00:00;resman;automaticsolver;True
0:00:15;resman;pump-3-state;failure
0:00:25;resman;pump-3-state;off
0:00:25;resman;automaticsolver;False
0:00:30;resman;pump-7-state;failure
0:00:40;resman;pump-7-state;off

# 3.c. Tracking events
# Schedule automatic solving
0:00:20;track;automaticsolver;True
0:00:40;track;automaticsolver;False


# 3.d. Communications events
# /!\ Be careful to let sufficient time during two prompts to avoid sound overlapping
# Schedule some radio prompts
0:00:00;communications;radioprompt;own
0:00:25;communications;radioprompt;other
0:00:40;communications;automaticsolver;True
0:00:45;communications;radioprompt;own


# 4. End tasks at 1 minute
0:01:00;resman;stop
0:01:00;track;stop
0:01:00;sysmon;stop
0:01:00;communications;stop
0:01:00;scheduling;stop
```
