OpenMATB comes with a simple text-based configuration file, located at the root of the repository (`config.ini`), that allows you to change the following parameters:

- `language`: locale to use (must be defined in the `locales` folder`), see the <a href="https://github.com/juliencegarra/OpenMATB/wiki/Internationalization">internationalization page</a>.
- `screen_index`: if multiple screens are used, index of the screen where to display OpenMATB (default=0)
- `font_name`: if you don't like the default font, feel free to specify a different one below. The font must be available on the system. Leave it empty to get your system default font.
- `fullscreen`: should openMATB be displayed in fullscreen (default=True)
- `scenario_path`: path of the scenario to run (e.g., Parasuraman_et_al_1993/high_reliability_block.txt)
- `display_session_number`: should the unique session ID be displayed at start? (default=True)
- `hide_on_pause`: should the OpenMATB environment be hidden when a pause is requested? (default=False)
- `clock_speed`: the relative speed of the main clock. Do not change its default value (1.0), unless for debug purpose.
- `highlight_aoi`: if True, OpenMATB will display a red frame around each widget, as well as its name. Can be useful for debug purpose.
