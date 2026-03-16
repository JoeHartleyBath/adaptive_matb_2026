First of all, notice that all the different plugins inherit from a common class, which gives access to the following three parameters:

| Variable name | Description | Values |
|---------------|-------------|--------|
| title         | Title of the task, displayed if the plugin is visible                                                                                                                                                                                        | System monitoring                                             |
| taskplacement          | Task location in a 3x2 canvas                                                                                                                                                                                              | topleft, topmid, topright, bottomleft, bottommid, bottomright |
| taskupdatetime         | Delay between task updates in milliseconds.                                                                                                                                                                                | 200                                                           |


## List of plugins
 * [System monitoring](#system-monitoring-sysmon)
 * [Tracking](#tracking-track)
 * [Scheduling view](#scheduling-view-scheduling)
 * [Communications](#communications-communications)
 * [Resources management](#resources-management-resman)
 * [Performance Viewer](#performance-viewer-performance)
 * [Parallel port](#parallel-port-parallelport)
 * [Lab streaming layer](#lab-streaming-layer-labstreaminglayer)
 * [Instructions](#instructions-instructions)
 * [Generic scales](#generic-scales-genericscales)

### System monitoring (`sysmon`)

| Variable name          | Description                                                                                                                                                                                                                | Values                                                        |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| title                  | Characters displayed above the task                                                                                                                                                                                        | System monitoring                                             |
| taskplacement          | Task location in a 3x2 canvas                                                                                                                                                                                              | topleft, topmid, topright, bottomleft, bottommid, bottomright |
| taskupdatetime         | Delay between task updates in milliseconds.                                                                                                                                                                                | 200                                                           |
| alerttimeout           | Maximum duration of a target event in milliseconds.                                                                                                                                                                        | 5000                                                          |
| automaticsolver        | When True, any failure will be automatically corrected after an automaticsolverdelay duration.                                                                                                                             | True/False                                                    |
| automaticsolverdelay   | Delay between a failure onset and its automatic correction, in milliseconds.                                                                                                                                               | 1000                                                          |
| displayautomationstate | If True, the current automation state (MANUAL vs AUTO ON) of the task is displayed.                                                                                                                                        | True/False                                                    |
| allowanykey            | If True, the subject can use any system monitoring key, to signal a failure. For instance, he or she could use the F2 key to signal a failure for the F4 gauge.                                                            | True/False                                                    |
| scalesnumofboxes       | Number of boxes in the vertical scales. Must be odd to admit a middle.                                                                                                                                                     | 11                                                            |
| safezonelength         | Amplitude of cursor movements around the scale center when there are no failure.                                                                                                                                           | 3                                                             |
| scalestyle             | Define the general appearance of the scales (MATB-I vs MATB-II)                                                                                                                                                            | 1/2                                                           |

#### Lights dictionary
Lights parameters are accessed through a variable hierarchy. Each light button (1 or 2) admits the parameters presented in the table below. As an example, if one wishes to alter the hexadecimal color value of the first light button, one should call the variable as lights-1-oncolor.


| Variable name | Description                                          | lights-1         | lights-2         | 
|---------------|------------------------------------------------------|------------------|------------------| 
| name          | String of characters displayed over the light button | F5               | F6               | 
| failure       | Is the gauge on failure currently ?                  | True/False       | True/False       | 
| on            | Is the gauge on [True] or off [False] ?              | True/False       | True/False       | 
| default       | What is the default state of the gauge ?             | on/off           | on/off           | 
| oncolor       | Hexadecimal color value of the light button          | #009900          | #FF0000          | 
| keys          | Response key associated with the gauge               | QtCore.Qt.Key_F5 | QtCore.Qt.Key_F6 | 

#### Scales dictionary
Each scale gauge (1, 2, 3 or 4) admits the parameters presented in the table below. As an example, if one wishes to cause a failure for the third scale, one should call the variable as scales-3-failure.

| Variable name | Description                                                           | scales-1         | scales-2         | scales-3         | scales-4         | 
|---------------|-----------------------------------------------------------------------|------------------|------------------|------------------|------------------| 
| name          | String of characters displayed over the light button                  | F1               | F2               | F3               | F4               | 
| failure       | Is the gauge on failure currently ? If yes, what is the target area ? | no/up/down       | no/up/down       | no/up/down       | no/up/down       | 
| keys          | Response key associated with the scale (see online documentation)     | QtCore.Qt.Key_F1 | QtCore.Qt.Key_F2 | QtCore.Qt.Key_F3 | QtCore.Qt.Key_F4 | 

#### Feedbacks dictionary

Feedbacks parameters are accessed through a variable hierarchy. Each feedback type (positive or negative) admits the parameters presented in the table below. As an example, if one wishes to alter the hexadecimal color value of the negative feedback, one should call the variable as feedbacks-negative-color.

| Variable name | Description                                                           | feedbacks-positive         | feedbacks-negative         | 
|---------------|-----------------------------------------------------------------------|------------------|------------------|
| active          | Is this type of feedback allowed ?    | True/False  | True/False | 
| color           | Defines the color of the feedback     | #ffff00       | #ff0000  |
| duration        | Duration in milliseconds, during which positive feedback is displayed after a correct response had been given. Note that a positive feedback is also displayed when the response is given by mean of the automatic solver.  | 1500 | 1500 |
| trigger         | If you wish to manually trigger a feedback to a given scale, just turn this parameter to the target scale index (1, 2, 3, 4). Default is 0. After the feedback has been displayed, this value will automatically turn back to default.  |  0  |  0  |


### Tracking (`track`)

| Variable name          | Description                                                                                                                                       | Values                                                        | 
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------| 
| title                  | Characters displayed above the task                                                                                                               | Tracking                                                      | 
| taskplacement          | Task location in a 3+ù2 canvas                                                                                                                     | topleft, topmid, topright, bottomleft, bottommid, bottomright | 
| taskupdatetime         | Delay between task updates in milliseconds.                                                                                                       | 50                                                            | 
| cursorcolor            | Hexadecimal value that defines the cursor color.                                                                                                  | #0000FF                                                       | 
| cursorcoloroutside     | Hexadecimal value that defines the cursor color when it is outside the target area.                                                               | #0000FF                                                       | 
| automaticsolver        | Is the tracking task managed by an automatism ?                                                                                                   | True/False                                                    | 
| assistedsolver         | Is the tracking task assisted by an automatism ? Assisted means that any joystick input from the participant leads the cursor toward the center.  | True/False                                                    | 
| displayautomationstate | If True, the current automation state (MANUAL vs AUTO ON vs ASSIST ON) of the task is displayed.                                                  | True/False                                                    | 
| targetradius           | Radius proportion of the target area. 0.1 means that the radius of the target area is 10% of the task total width. 0 means no target area at all. | 0.1                                                           | 
| joystickforce          | The smaller this factor, the more the joystick movement is attenuated. Greater values leads to a more sensitive joystick.                         | 0.05                                                          | 
| cutofffrequency        | Cutoff frequency of the sinusoides, that are summed to generate the tracking cursor movements.                                                    | 0.06                                                          | 
| equalproportions       | If True, vertical and horizontal axes are the same size. If False, the horizontal axis extends over the entire available width.                   | True/False                                                    | 

### Scheduling view (`scheduling`)

| Variable name  | Description                                 | Values                                                        | 
|----------------|---------------------------------------------|---------------------------------------------------------------| 
| title          | Characters displayed above the task         | Scheduling                                                    | 
| taskplacement  | Task location in a 3+ù2 canvas               | topleft, topmid, topright, bottomleft, bottommid, bottomright | 
| taskupdatetime | Delay between task updates in milliseconds. | 1000                                                          | 

### Communications (`communications`)

| Variable name          | Description                                                                                                                                                                                                                                                                     | Values                                                        | 
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------| 
| title                  | Characters displayed above the task                                                                                                                                                                                                                                             | Communications                                                | 
| taskplacement          | Task location in a 3+ù2 canvas                                                                                                                                                                                                                                                   | topleft, topmid, topright, bottomleft, bottommid, bottomright | 
| taskupdatetime         | Delay between task updates in milliseconds.                                                                                                                                                                                                                                     | 50                                                            | 
| callsignregex          | Regular expression pattern for callsign generation.                                                                                                                                                                                                                             | [A-Z][A-Z][A-Z]\d\d\d                                         | 
| owncallsign            | Callsign of the subject. If empty, automatically generated according to callsignregex.                                                                                                                                                                                          | (empty)                                                       | 
| othercallsignnumber    | Number of irrelevant distracting callsigns.                                                                                                                                                                                                                                     | 5                                                             | 
| othercallsign          | List of distracting callsigns. If empty, automatically generated according to callsignregex and othercallsignnumber.                                                                                                                                                            | (empty)                                                       | 
| airbandminMhz          | Minimum radio frequency, in Mhz.                                                                                                                                                                                                                                                | 108                                                           | 
| airbandmaxMhz          | Maximum radio frequency, in Mhz.                                                                                                                                                                                                                                                | 137                                                           | 
| targetrangedistanceMhz | Defines the distance range in which the target frequency is selected, as a function of the current frequency. Input is a list. For instance, [5,6] argument leads the target frequency to be comprised at a distance from 5 to 6 Mhz of the current frequency (above or below). | [5,6]                                                         | 
| radiostepMhz           | One left/right key press leads to a decrease/increase of one radiostepMhz, if possible.                                                                                                                                                                                         | 0.1                                                           | 
| voiceidiom             | Voice idiom. The corresponding folder must be present in the Sound directory.                                                                                                                                                                                                   | english/french                                                | 
| voicegender            | Voice gender. The corresponding folder must be present in the selected idiom directory.                                                                                                                                                                                         | female/male                                                   | 
| radioprompt            | Used to initiate either a target (own) or a distractor (other) prompt [own or other]. Initially empty. Will be emptied at the end of the prompt.                                                                                                                                | (empty)/own/other                                             | 
| promptlist             | List of radio labels, in their order of appearance. Each corresponding file must be available in the sound folder. This list is used both for target and distractors radio lists.                                                                                               | [NAV1, NAV2, COM1, COM2]                                      | 
| automaticsolver        | Is the communication task managed by an automatism ?                                                                                                                                                                                                                            | True/False                                                    | 
| displayautomationstate | If True, the current automation state (MANUAL vs AUTO ON) of the task is displayed.                                                                                                                                                                                             | True/False                                                    | 

### Resources management (`resman`)

| Variable name                                                                                                                                  | Description                                                                                           | Values                                                        | 
|------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------| 
| title                                                                                                                                          | Characters displayed above the task                                                                   | Resources manager                                             | 
| taskplacement                                                                                                                                  | Task location in a 3+ù2 canvas                                                                         | topleft, topmid, topright, bottomleft, bottommid, bottomright | 
| taskupdatetime                                                                                                                                 | Delay between task updates in milliseconds.                                                           | 2000                                                          | 
| heuristicsolver                                                                                                                                | If True, the task is managed by an automatism which relies on the three following heuristics : (1) systematically activate pumps draining non-depletable tanks; (2) activate/deactivate pump whose target tank is too low/high (""Too"" means level is out of a tolerance zone around the target level, see below); (3) equilibrate between the two A/B tanks if one tank is full enough to feed the other                                                           | True/False                                                                                            |                                                               | 
| assistedsolver                                                                                                                                 | If True, the task is managed by an the heuristic solver (see above) but does not block manual inputs. | True/False                                                    | 
| displayautomationstate                                                                                                                         | If True, the current automation state (MANUAL vs AUTO ON vs ASSIST ON) of the task is displayed.      | True/False                                                    | 
| tolerancelevel                                                                                                                                 | Absolute distance to target level, below which performance may be considered acceptable.              | 500                                                           | 
| displaytolerance                                                                                                                               | Should the tolerance level be displayed ?                                                             | True/False                                                    | 
| pumpcoloroff                                                                                                                                   | Hexadecimal value coding the pump color when off                                                      | #AAAAAA                                                       | 
| pumpcoloron                                                                                                                                    | Hexadecimal value coding the pump color when on                                                       | #00FF00                                                       | 
| pumpcolorfailure                                                                                                                               | Hexadecimal value coding the pump color when on failure                                               | #FF0000                                                       | 

#### Pump dictionary
Each pump (1, 2, ÔÇª, 8) admits the parameters presented in the table below. As an example, if one wishes to change flow for the fifth pump, one should call the variable as pump-5-flow. Be careful, modifying fromtank or totank variables wonÔÇÖt affect the graphical representation which is static for the moment.

| Variable name | Description                             | pump-1          | pump-2          | pump-3          | pump-4          | pump-5          | pump-6          | pump-7          | pump-8          | 
|---------------|-----------------------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------| 
| flow          | Volume transferred per minute.          | 800             | 800             | 600             | 600             | 600             | 600             | 400             | 400             | 
| state         | Current pump state 0=off, 1=on, -1=fail | 0/1/-1          | 0/1/-1          | 0/1/-1          | 0/1/-1          | 0/1/-1          | 0/1/-1          | 0/1/-1          | 0/1/-1          | 
| keys          | Response key associated with each pump  | QtCore.Qt.Key_1 | QtCore.Qt.Key_2 | QtCore.Qt.Key_3 | QtCore.Qt.Key_4 | QtCore.Qt.Key_5 | QtCore.Qt.Key_6 | QtCore.Qt.Key_7 | QtCore.Qt.Key_8 | 
| hide          | Is this pump hidden ?                   | 0/1             | 0/1             | 0/1             | 0/1             | 0/1             | 0/1             | 0/1             | 0/1             | 

####-áTank dictionary
Each tank (a, b, c, d, e, f) admits the parameters presented in the table below. As an example, if one wishes to change the maximum level for the d tank, one should call the variable as tank-d-max.

| Variable name | Description                                              | tank-a | tank-b | tank-c | tank-d | tank-e | tank-f | 
|---------------|----------------------------------------------------------|--------|--------|--------|--------|--------|--------| 
| level         | Tank level                                               | 2500   | 2500   | 1000   | 1000   | 3000   | 3000   | 
| max           | Maximum tank level                                       | 4000   | 4000   | 2000   | 2000   | 4000   | 4000   | 
| target        | Optimal level to reach                                   | 2500   | 2500   | None   | None   | None   | None   | 
| depletable    | Is the tank depletable ? (zero means unlimited capacity) | 1      | 1      | 1      | 1      | 0      | 0      | 
| lossperminute | Volume that is lost each minute                          | 800    | 800    | 0      | 0      | 0      | 0      | 
| hide          | Is this tank hidden ?                                    | 0/1    | 0/1    | 0/1    | 0/1    | 0/1    | 0/1    | 


### Resources management pump status (`pumpstatus`)

| Variable name  | Description                                 | Values                                                        | 
|----------------|---------------------------------------------|---------------------------------------------------------------| 
| title          | Characters displayed above the task         | Pumps                                                         | 
| taskplacement  | Task location in a 3+ù2 canvas               | topleft, topmid, topright, bottomleft, bottommid, bottomright | 
| taskupdatetime | Delay between task updates in milliseconds. | 1000                                                          | 
