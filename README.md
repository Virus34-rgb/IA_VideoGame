# IA_VideoGame

========== CUMULATIVE ==========
Mon Aug 31 05:07:13 2026    profile_replay_optimizedRunLarga.prof

         2482601573 function calls (2364162965 primitive calls) in 14699.363 seconds

   Ordered by: cumulative time
   List reduced from 17709 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    20003   10.835    0.001 76486.342    3.824 C:\Users\josel\AppData\Local\Python\pythoncore-3.14-64\Lib\asyncio\base_events.py:1977(_run_once)
    27100    2.616    0.000 18169.233    0.670 C:\Users\josel\AppData\Local\Python\pythoncore-3.14-64\Lib\asyncio\events.py:92(_run)
        1    0.000    0.000 14694.115 14694.115 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\wandb\sdk\lib\asyncio_manager.py:264(_main)
     6330    0.413    0.000 14244.183    2.250 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\wandb\sdk\lib\asyncio_manager.py:227(_wrap)
  1200000  140.431    0.000 9084.858    0.008 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\playerAIV.py:97(replay_turn)
    20003   30.732    0.002 8946.100    0.447 C:\Users\josel\AppData\Local\Python\pythoncore-3.14-64\Lib\asyncio\windows_events.py:444(select)
     1000   15.033    0.015 4997.580    4.998 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\trainerV.py:429(_replay_turn_and_selection)
  1800000   14.579    0.000 4219.185    0.002 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\playerAIV.py:297(_optimize_step)
      600    0.284    0.000 3690.515    6.151 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\trainerV.py:165(_run_batch)
    20002   29.552    0.001 3322.117    0.166 C:\Users\josel\AppData\Local\Python\pythoncore-3.14-64\Lib\asyncio\windows_events.py:765(_poll)
     1000   28.746    0.029 3241.544    3.242 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\trainerV.py:444(_remember_and_replay_selection_batch)
   600000   65.629    0.000 3078.421    0.005 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\playerAIV.py:65(replay_selection)
53820180/7362738  104.674    0.000 2729.906    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\nn\modules\module.py:1774(_wrapped_call_impl)
53820180/7362738  168.560    0.000 2705.982    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\nn\modules\module.py:1782(_call_impl)
  1800000   42.813    0.000 2381.956    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\optim\optimizer.py:509(wrapper)
  1800000   23.384    0.000 2208.753    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\optim\optimizer.py:60(_use_grad)
  1800000   23.933    0.000 2172.854    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\optim\adam.py:214(step)
  3759138  249.909    0.000 1994.184    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\turnNetwork.py:33(forward)
  1800000    8.246    0.000 1914.697    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\optim\optimizer.py:131(maybe_fallback)
  1800000   44.600    0.000 1904.871    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\optim\adam.py:902(adam)
  1800000  105.133    0.000 1837.663    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\optim\adam.py:553(_multi_tensor_adam)
  1200000  184.840    0.000 1818.838    0.002 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\playerAIV.py:305(_multi_agent_double_dqn_target)
    12600    2.083    0.000 1728.174    0.137 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\trainerV.py:221(_run_turn)
  1800000    8.144    0.000 1517.866    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\_tensor.py:566(backward)
  1800000   35.005    0.000 1509.222    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\autograd\__init__.py:255(backward)
 26010090   64.545    0.000 1412.932    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\nn\modules\linear.py:130(forward)
  1800000   19.230    0.000 1401.036    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\autograd\graph.py:966(_engine_run_backward)
  1800000 1364.863    0.001 1364.863    0.001 {method 'run_backward' of 'torch._C._EngineBase' objects}
19822584/7807692   69.830    0.000 1138.806    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\utils\_contextlib.py:120(decorate_context)
  2559138  685.403    0.000 1081.986    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\playerAIV.py:229(mask_turn)
  1200000  105.036    0.000 1052.764    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\replayMemoryAN.py:68(sample)
 26010090  952.088    0.000  952.088    0.000 {built-in method torch._C._nn.linear}
   159138   22.088    0.000  922.935    0.006 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\playerAIV.py:205(turn)
  6007446   49.436    0.000  901.696    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\noisy_linear.py:112(reset_noise)
  3759138   47.191    0.000  881.920    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\nn\modules\container.py:248(forward)
    12600   14.889    0.001  858.982    0.068 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\trainerV.py:536(_turn_mixed_opponent)
  2402400    9.485    0.000  777.731    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\turnNetwork.py:51(reset_noise)
  1824000  277.735    0.000  756.405    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\sumTree.py:16(_propagate_batch)
  1800000  703.895    0.000  739.453    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\sumTree.py:71(get_batch)
  1800000   51.357    0.000  729.728    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\sumTree.py:63(update_batch)



========== TOTTIME ==========
Mon Aug 31 05:07:13 2026    profile_replay_optimizedRunLarga.prof

         2482601573 function calls (2364162965 primitive calls) in 14699.363 seconds

   Ordered by: internal time
   List reduced from 17709 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  1800000 1364.863    0.001 1364.863    0.001 {method 'run_backward' of 'torch._C._EngineBase' objects}
 26010090  952.088    0.000  952.088    0.000 {built-in method torch._C._nn.linear}
  1800000  703.895    0.000  739.453    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\sumTree.py:71(get_batch)
  2559138  685.403    0.000 1081.986    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\playerAIV.py:229(mask_turn)
  1800000  530.371    0.000  535.085    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\replayStorage.py:61(get_batch)
 29763015  462.190    0.000  462.190    0.000 {method 'at' of 'numpy.ufunc' objects}
  3600000  303.811    0.000  303.811    0.000 {built-in method torch._foreach_add_}
   601610  299.869    0.000  300.458    0.000 {built-in method _io.open}
   162932  295.005    0.002  295.005    0.002 {built-in method torch.multinomial}
  1824000  277.735    0.000  756.405    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\sumTree.py:16(_propagate_batch)
 12955390  272.352    0.000  272.352    0.000 {built-in method torch.where}
  3759138  249.909    0.000 1994.184    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\turnNetwork.py:33(forward)
  1800000  234.586    0.000  234.586    0.000 {built-in method torch._foreach_mul_}
  1800000  225.912    0.000  225.912    0.000 {built-in method torch._foreach_div_}
  1800000  211.704    0.000  211.704    0.000 {built-in method torch._foreach_sqrt}
 16688214  193.902    0.000  193.902    0.000 {built-in method torch.relu}
  9321876  190.934    0.000  209.141    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\noisy_linear.py:129(weight)
  1200000  184.840    0.000 1818.838    0.002 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\playerAIV.py:305(_multi_agent_double_dqn_target)
  8578338  183.121    0.000  183.121    0.000 {method 'gather' of 'torch._C.TensorBase' objects}
 10325328  180.966    0.000  180.966    0.000 {method 'sum' of 'torch._C.TensorBase' objects}
53820180/7362738  168.560    0.000 2705.982    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\nn\modules\module.py:1782(_call_impl)
  1800000  147.307    0.000  147.307    0.000 {built-in method torch._foreach_addcdiv_}
  9321876  144.872    0.000  165.365    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\noisy_linear.py:137(bias)
  1200000  140.431    0.000 9084.858    0.008 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\playerAIV.py:97(replay_turn)
  5457632  139.356    0.000  139.356    0.000 {method 'any' of 'torch._C.TensorBase' objects}
   603972  136.254    0.000  136.254    0.000 {method '__exit__' of '_io._IOBase' objects}
  1800000  132.587    0.000  216.439    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\optim\adam.py:139(_init_group)
157475250  130.707    0.000  130.707    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\nn\modules\module.py:1954(__getattr__)
 10721069  124.360    0.000  124.360    0.000 {method 'float' of 'torch._C.TensorBase' objects}
  1800000  114.180    0.000  114.180    0.000 {built-in method torch._foreach_lerp_}
  1800000  110.566    0.000  110.566    0.000 {built-in method torch._foreach_addcmul_}
  3600000  105.537    0.000  105.537    0.000 {built-in method torch._ops.profiler._record_function_enter_new}
  1800000  105.133    0.000 1837.663    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\optim\adam.py:553(_multi_tensor_adam)
  1200000  105.036    0.000 1052.764    0.001 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\AI\Agent\replayMemoryAN.py:68(sample)
53820180/7362738  104.674    0.000 2729.906    0.000 C:\Users\josel\OneDrive\Escritorio\Ángel\Desarrollo\IA_VideoGame\.venv\Lib\site-packages\torch\nn\modules\module.py:1774(_wrapped_call_impl)
  4801636  103.471    0.000  103.471    0.000 {built-in method torch.tensor}
 12014892  101.892    0.000  101.892    0.000 {built-in method torch.randn}
  1800000   98.296    0.000   98.296    0.000 {built-in method torch._C._group_tensors_by_device_and_dtype}
 14857263   89.369    0.000   89.369    0.000 {method 'view' of 'torch._C.TensorBase' objects}
  6007446   88.816    0.000   88.816    0.000 {method 'outer' of 'torch._C.TensorBase' objects}