OpenMATB offers the possibility to present and record responses from a questionnaire, at any desired time. Available questionnaires can be found in the `Scales` folder.

## Present a questionnaire
To present a questionnaire during the experiment, all one has to do is adding the corresponding statement at the desired time, in the relevant scenario file. For instance, if the english version of the Nasa-TLX has to be presented after 1 minute of multitasking, one should add the following to the scenario : 
```
# Scale name can be stated at any time (before it is presented)
0:00:00;genericscales;filename;nasatlx_eng.txt
0:01:00;genericscales;start
```

## Create a custom scale
For the need of the experiment, one might want to create a custom scale. To do so : 
1. Copy-paste the scale_template_en.txt
2. Rename it like you want
3. Complete the file with the scale items, following the given template (see below)

Each item of the scale must be formatted like the following

`Short_title;Question;minimumName/maximumName;minimumValue/maximumValue/defaultValue`

(`minimumName` and `maximumName` are the two labels that limit the scale, e.g., LOW and HIGH)

```
# This is a scale template

# For each item, you must indicate the question, the name of the minimal scale value (e.g., Very Low),
the name of the maximal scale value (e.g., Very high), as well as the minimum, maximum
and default value for each question.

# Each item must be formatted like the following :
# Short_title;Question;minimumName/maximumName;minimumValue/maximumValue/defaultValue
# minimumName and maximumName are the two labels that limit the scale

# Item 1
Short title;Question;Minimum/Maximum;0/50/25

#-·Item 2
Short title;Question;Minimum/Maximum;0/50/25

# ...

```

**For now, the program retrieves information from a scale file (e.g., `nasatlx_eng.txt`) and attempts to display all scales at one time. If you want to unpack your items on two or more screens, you must explicitly create and run the equivalent number of questionnaires.**
