Cada sesión comienza con un breve wizard donde se deben indicar varias cosas:
- El modo en el que se ejecuta: autónomo, sugerencias o solo observando.
- El tipo de decisor a usar, en caso de no haber elegido modo observador
- El objetivo a realizar. Éste se debe escoger de un catálogo de objetivos, algunos de los cuales requieren de indicar otros detalles concretos, como posibles targets.
- El nombre de la sesión (utilizada para dar nombre a los directorios donde guarda su info)
- El nombre la empresa u objetivo (puede ser el mismo de la sesión)

Una vez arranca la sesión:
- Primero se crean el context/foco y la KB
- Con esto comienza la primera iteración:
    - A partir de estos se crear el primer estado
    - Este estado será la entrada del decisor, que decidirá una acción
    - A partir de esta acción se forma el comando a ejecutar (si es necesario, se utilizará el context para completar placeholders si los hay)
    - Al ejecutarse el comando, se obtiene una salida, que debe ser extraida, procesada y obtenerse info estructurada, estos son las evidencias
    - Estas evidencias se asocian a unos eventos.
    - Ahora es monento de actualizar la KB y el context, utilizando los eventos sabe dónde guardar las evidencias.
    - Además, se guarda en log toda la info de este step o iteración
- Segunda iteración
    - Con el context/foco y KB actualizados, se generá un nuevo estado 
    - Se repiten todos los pasos.
    - Si la acción elegida es la 0 (STOP), se acaba la sesión

- La información recabada en la KB será usada para generar, opcionalmente, un informe de pentesting