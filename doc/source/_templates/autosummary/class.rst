{{ fullname | escape }}

{% if objname == '__init__' %}
.. automethod:: {{ fullname }}
   :noindex:
{% else %}
.. autoclass:: {{ fullname }}
   :members:
   :undoc-members:
{% endif %}
