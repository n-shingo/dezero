# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:06:18 2024

@author: shingo
"""
is_simple_core = False

if is_simple_core :
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable

else:
    from dezero.core import Variable
    from dezero.core import Function
    from dezero.core import using_config
    from dezero.core import no_grad
    from dezero.core import as_array
    from dezero.core import as_variable
    from dezero.core import setup_variable
    
    from dezero.layers import Layer
    from dezero.models import Model

    import dezero.functions
    import dezero.layers
    import dezero.models
    
setup_variable()