This repository contains the code used to train, evaluate, and plot results for solving the Cart-pole Balance task from the DeepMind Control Suite (DMC) using Proximal Policy Optimization (PPO).
Training is performed with three seeds (0, 1, 2) and evaluation is performed using seed = 10. Code Was executed in Windows OS.


Start by Cloning the repo. Then.

Step 1: Install Packages

-> pip install dm_control torch numpy matplotlib opencv-python

Note - Any other insufficient packages the terminal suggests.

Step 2: Train the PPO Agent (Seeds 0, 1, 2)

-> python Train.py --seed 0
-> python Train.py --seed 1
-> python Train.py --seed 2

Will generate training_returns_seedX.npy, ppo_model_seedX.pth for each seed. These are the weights and the results.

Note - Simulation has been commented off as it takes longer time to train. Uncomment the following part for simulation:

            """ UNCOMMENT FOR SIMULATION
            # Rendering Live Simulation 
            frame = env.physics.render(height=480, width=640, camera_id=0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(f"PPO Training Seed {seed}", frame)

            # To Quit simulation
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print("Stopped training manually.")
                return
            """

Step 3: Evaluate the Trained Models (Seed 10)

-> python Evaluation.py --model ppo_model_seed0.pth
-> python Evaluation.py --model ppo_model_seed1.pth
-> python Evaluation.py --model ppo_model_seed2.pth

To view the simulation during evaluation, add:

python eval_ppo_seed10.py --model ppo_model_seed0.pth --render

Each evaluation run will generate:

ppo_model_seedX_eval_seed10.npy


Step 4: Plot Training and Evaluation Curves

After generating all .npy result files, run:

-> python Plot.py 

