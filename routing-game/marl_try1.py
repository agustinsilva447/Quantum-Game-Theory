from pygeneses.envs.prima_vita import PrimaVita

params_dic = {"initial_population": 20, "state_size": 32, "speed": 30, "model_updates": 5}
model = PrimaVita(mode="human", params_dic=params_dic)
model.run(stop_at=100)