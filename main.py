'''
Main program file.

func:
- train_ppo
- train_TCSAC
- test
- baseline
'''


from TCSAC_controller import TCSAC


if __name__ == '__main__':

    tscac = TCSAC()
    tscac.train()   
    # --- configurations ---
  