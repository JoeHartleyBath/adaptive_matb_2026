You should read <a href="https://github.com/juliencegarra/OpenMATB/wiki/How-to-build-a-scenario-file">how to write correctly a scenario file</a> first.

## Presentation of the `instructions` plugin.

OpenMATB offers the possibility to display written instructions at desired time. The functioning of this plugin slightly differs from the task plugins, since, in the scenario script, information of the plugin must be set before the plugin is started. For instance, suppose you saved your instruction in the `includes/instructions/myinstructions.txt`, and you want to display it at start. You would do the following:

```
# Starting instruction
0:00:00;instructions;filename;myinstructions.txt  # It is define *before* plugin start
0:00:00;instructions;start                        # So the plugin knows what to display when started

0:00:00;sysmon;start
0:00:00;track;start
0:00:00;scheduling;start
0:00:00;resman;start
0:00:00;communications;start

[...]
```

Starting the `instructions` plugin will automatically `pause` and `hide` all the other (task) plugins. These task will be automatically resumed, and showed when the instructions are finished.

What is fundamental with instructions is that they must be formatted in a simple HTML fashion. This allow you to use some HTML tags like `<h1>`, `<h2>`, `<p>`, `<center>`, `<em>`, `<strong>`. [Pyglet documentation on github](https://github.com/pyglet/pyglet/blob/master/pyglet/text/formats/html.py) mentions that the following attributes are fully supported : B BLOCKQUOTE BR CENTER CODE DD DIR DL EM FONT H1 H2 H3 H4 H5 H6 I IMG KBD LI MENU OL P PRE Q SAMP STRONG SUB SUP TT U UL VAR.

Moreover, HTML styling allows you to display images in the body of your instructions. For instance, if you want to display an image of the OpenMATB environment, along with instructions, just insert `<p><img src=openmatb.png></p>`. The `src` image must be available in the `includes/img` folder. No image resizing feature is available, so be careful to size your image by hand before using it.

If you want your instructions to be displayed over multiple screens, just add the `<newpage>` tag where relevant in your instructions.

Note that you can also use the `maxdurationsec` parameter to define a time limit to instructions reading. Along with the `allowkeypress` parameter (set to False), you can impose instructions to be displayed for the desired time (be careful that this is not compatible with splitting one instruction into several screens).

## List of `instructions` parameters

|Variable|Description|Possible values|Default|
|:----|:----|:----|:----|
|title|Title of the task, displayed if the plugin is visible|(string)|Instructions|
|taskplacement|Task location in a 3x2 canvas|`topleft`, `topmid`, `topright`, `bottomleft`, `bottommid`, `bottomright`, `fullscreen`|fullscreen|
|taskupdatetime|Delay between plugin updates (ms)|(positive integer)|15|
|filename|Name of the instructions file to use|(string)|(empty)|
|maxdurationsec|Maximum displaying duration (seconds)|(positive integer)|0|
|response-text|String chain to prompt the subject to press a continuing key|(string)|'Press SPACE to continueÔÇÖ|
|response-key|Keyboard key to continue|(keyboard key)|space|


