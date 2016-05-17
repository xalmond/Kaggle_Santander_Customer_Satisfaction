# Kaggle - Banco Santander: Customer satisfaction
https://www.kaggle.com/c/santander-customer-satisfaction

Desde los equipos de soporte hasta la línea de dirección, para todos los estamentos del Banco Santander la satisfacción del cliente es una medida clave del éxito. Por lo tanto el Banco Santander es consciente que un cliente insatisfecho, no sólo no seguirá siéndolo, sino que además abandonará sin avisar.

![](https://github.com/xalmond/kaggle_santander_customer_satisfaction/blob/master/submission/images/santander_custsat_red.jpg)  

Por ello, con fecha del 2 de Marzo de 2016, publica en la plataforma [Kaggle](http://www.kaggle.com) una competición con título ["¿Qué clientes son clientes felices?](https://www.kaggle.com/c/santander-customer-satisfaction). Mediante dicha competición, el banco solicita la ayuda a todos los Kagglers para conseguir identificar lo antes posible clientes insatisfechos, y así poder anticipar las acciones necesarias para mejorar su satisfacción antes que sea demasiado tarde.

### Descripción de los directorios

El directorio [real](https://github.com/xalmond/kaggle_santander_customer_satisfaction/tree/master/real) contiene los scripts de R necesarios para responder a la competición, y están basados en el modelo XGBoost - e**X**treme **G**radient **B**oost:

    + Lectura y preparación de los datasets
    + Selección de variables
    + Cross-validation
    + Predicción

El directorio [kschool](https://github.com/xalmond/kaggle_santander_customer_satisfaction/tree/master/kschool) contiene la adaptación de los scripts e información adicional como documentación del proyecto fin del [Máster de Data Science de 2016 en KSchool](http://kschool.com/cursos/madrid/master-en-data-science/).
