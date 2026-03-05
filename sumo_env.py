import os
import sys
import numpy as np

# Check for SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    # Use the Eclipse SUMO installation path
    sumo_tools_path = "/Library/Frameworks/EclipseSUMO.framework/Versions/Current/EclipseSUMO/tools"
    if os.path.exists(sumo_tools_path):
        sys.path.append(sumo_tools_path)
    else:
        # Fallback to homebrew path
        sys.path.append("/opt/homebrew/opt/sumo/share/sumo/tools")

import traci

import config


class SumoEnvironment:
    """
    SUMO Environment wrapper for DQN traffic light control.
    Supports dynamic flow file switching and traffic metrics collection.
    Supports named TraCI connections for parallel worker processes.
    """
    
    def __init__(self, use_gui=False, worker_id=None):
        self.use_gui = use_gui
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.flow_file = None  # Will be set dynamically
        
        # Worker identification for parallel training
        self.worker_id = worker_id
        self.traci_label = f"worker_{worker_id}" if worker_id is not None else "default"
        self.conn = None  # TraCI connection reference
        
        self.tls_ids = list(config.TLS_IDS.values())
        self.current_phase = 0
        self.last_phase = 0
        self.time_since_last_action = 0
        self.step_count = 0
        
        # Will be populated after first reset
        self.incoming_lanes = []
        self.outgoing_lanes = []
        self.state_size = None
        
        # Metrics collection for current episode
        self.episode_queue_lengths = []
        self.episode_waiting_times = []
    
    def set_flow_file(self, flow_file):
        """
        Set the flow file to use for the next simulation.
        Must be called before reset().
        
        Args:
            flow_file: Path to the flow XML file (e.g., 'Enviroment/flows_day_1.xml')
        """
        self.flow_file = flow_file
        
    def _build_sumo_cmd(self, start_time=0):
        """Build the SUMO command with the appropriate flow file and start time."""
        cmd = [self.sumo_binary, "-c", config.SUMO_CONFIG_PATH, "--no-warnings", "--no-step-log"]
        
        if start_time > 0:
            cmd.extend(["--begin", str(start_time)])
            
        # If a specific flow file is set, override the route-files
        if self.flow_file:
            # We need to provide VType and routes along with the flow file
            route_files = f"Enviroment/VType.xml,Enviroment/routes.rou.xml,{self.flow_file}"
            cmd.extend(["--route-files", route_files])
        
        return cmd
        
    def _get_lanes(self):
        """
        Get all unique incoming and outgoing lanes for the traffic lights.
        
        Returns:
            incoming_lanes: Unique lanes entering the intersection (controlled by TLS)
            outgoing_lanes: Unique lanes exiting the intersection
        """
        incoming = set()
        outgoing = set()
        
        for tls_id in self.tls_ids:
            # getControlledLanes() returns duplicates (same lane for different links)
            # So we use set() to keep only unique lanes
            controlled_lanes = self.conn.trafficlight.getControlledLanes(tls_id)
            for lane in controlled_lanes:
                incoming.add(lane)
                
                # Get outgoing lanes from this lane's connections
                links = self.conn.lane.getLinks(lane)
                for link in links:
                    # link[0] is the outgoing lane
                    # Only add if it's not also an incoming lane (avoid internal lanes)
                    if link[0]:  # Check if link exists
                        outgoing.add(link[0])
        
        # Remove any overlap - a lane shouldn't be both incoming and outgoing
        # (This handles internal junction lanes that might appear in both sets)
        outgoing = outgoing - incoming
        
        return sorted(list(incoming)), sorted(list(outgoing))
    
    def _get_lane_vehicle_count(self, lane):
        """Get number of vehicles on a lane."""
        try:
            return self.conn.lane.getLastStepVehicleNumber(lane)
        except:
            return 0
    
    def _get_state(self):
        """
        Get current state representation.
        State = [lane_1_count, lane_2_count, ..., lane_n_count, phase_one_hot]
        """
        # Get vehicle counts for each lane
        lane_counts = []
        for lane in self.incoming_lanes:
            lane_counts.append(self._get_lane_vehicle_count(lane))
        for lane in self.outgoing_lanes:
            lane_counts.append(self._get_lane_vehicle_count(lane))
        
        # One-hot encode current phase
        phase_one_hot = [0] * config.NUM_ACTIONS
        phase_one_hot[self.current_phase] = 1
        
        state = np.array(lane_counts + phase_one_hot, dtype=np.float32)
        return state
    
    def _calculate_reward(self):
        """
        Calculate reward based on Total Waiting Time penalty.
        R = - (Total waiting time of all vehicles at the intersection) / 100
        This forces the agent to clear long-waiting vehicles to minimize the growing penalty.
        """
        total_wait = self.get_total_waiting_time()
        
        # We scale it down slightly so the neural network gradients don't explode
        reward = -total_wait / 100.0
        return reward
    
    def get_total_queue_length(self):
        """
        Get total queue length (number of halting vehicles) across all incoming lanes.
        A vehicle is considered halting if its speed is below 0.1 m/s.
        """
        total_halting = 0
        for lane in self.incoming_lanes:
            try:
                total_halting += self.conn.lane.getLastStepHaltingNumber(lane)
            except:
                pass
        return total_halting
    
    def get_total_waiting_time(self):
        """
        Get total waiting time across all incoming lanes.
        Returns total waiting time in seconds.
        """
        total_waiting = 0.0
        for lane in self.incoming_lanes:
            try:
                total_waiting += self.conn.lane.getWaitingTime(lane)
            except:
                pass
        return total_waiting
    
    def get_average_queue_length(self):
        """Get average queue length per incoming lane."""
        if not self.incoming_lanes:
            return 0.0
        return self.get_total_queue_length() / len(self.incoming_lanes)
    
    def get_average_waiting_time(self):
        """Get average waiting time per incoming lane."""
        if not self.incoming_lanes:
            return 0.0
        return self.get_total_waiting_time() / len(self.incoming_lanes)
    
    def _set_phase(self, phase_idx):
        """
        Set traffic lights according to the phase.
        First set all lights to red, then enable specific lights for the phase.
        """
        # Set all lights to red (phase 2 is typically all-red in SUMO)
        for tls_id in self.tls_ids:
            try:
                # Get current state and set all to red
                current_state = self.conn.trafficlight.getRedYellowGreenState(tls_id)
                red_state = 'r' * len(current_state)
                self.conn.trafficlight.setRedYellowGreenState(tls_id, red_state)
            except:
                pass
        
        # Now set specific lights to green based on phase
        if phase_idx == 0:  # PHASE_1_STRAIGHTS
            # Maitighar <-> Kupondole
            try:
                self.conn.trafficlight.setPhase(config.TLS_IDS["maitighar_main"], 0)
                self.conn.trafficlight.setPhase(config.TLS_IDS["kupondole"], 0)
            except:
                pass
                
        elif phase_idx == 1:  # PHASE_2_MAITIGHAR_RIGHT
            # Maitighar -> Tripureshwor + Kupondole
            try:
                self.conn.trafficlight.setPhase(config.TLS_IDS["maitighar_right"], 0)
                self.conn.trafficlight.setPhase(config.TLS_IDS["maitighar_main"], 0)
            except:
                pass
                
        elif phase_idx == 2:  # PHASE_3_TRIPURESHWOR_STRAIGHT
            # Tripureshwor -> Maternity + Left
            try:
                self.conn.trafficlight.setPhase(config.TLS_IDS["tripureshwor"], 0)
                self.conn.trafficlight.setPhase(config.TLS_IDS["maternity"], 0)
            except:
                pass
                
        elif phase_idx == 3:  # PHASE_4_KUPONDOLE_RIGHT
            # Kupondole -> Maternity + Maitighar
            try:
                self.conn.trafficlight.setPhase(config.TLS_IDS["kupondole"], 0)
            except:
                pass
    
    def _set_yellow_phase(self):
        """Set all traffic lights to yellow."""
        for tls_id in self.tls_ids:
            try:
                current_state = self.conn.trafficlight.getRedYellowGreenState(tls_id)
                yellow_state = 'y' * len(current_state)
                self.conn.trafficlight.setRedYellowGreenState(tls_id, yellow_state)
            except:
                pass
    
    def reset(self, start_time=0):
        """Reset the environment to a specific start time in the simulation."""
        # Close existing connection if any
        try:
            self.conn.close()
        except:
            pass
        
        # Build command with appropriate flow file and start time
        sumo_cmd = self._build_sumo_cmd(start_time)
        
        # Start new simulation with named connection label
        traci.start(sumo_cmd, label=self.traci_label)
        self.conn = traci.getConnection(self.traci_label)
        
        # Initialize lanes
        self.incoming_lanes, self.outgoing_lanes = self._get_lanes()
        
        # Calculate state size
        self.state_size = len(self.incoming_lanes) + len(self.outgoing_lanes) + config.NUM_ACTIONS
        
        # Reset phase and counters
        self.current_phase = 0
        self.last_phase = 0
        self.time_since_last_action = 0
        self.step_count = 0
        
        # Reset episode metrics
        self.episode_queue_lengths = []
        self.episode_waiting_times = []
        
        # Set initial phase
        self._set_phase(self.current_phase)
        
        return self._get_state()
    
    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).
        Also collects queue length and waiting time metrics.
        
        Args:
            action: Phase index (0-3)
        """
        # Determine if we need to switch phase
        phase_switch = (action != self.current_phase)
        
        # If switching, apply yellow phase for 3 seconds
        if phase_switch:
            self._set_yellow_phase()
            for _ in range(config.YELLOW_PHASE_DURATION):
                if self.conn.simulation.getMinExpectedNumber() <= 0:
                    break
                self.conn.simulationStep()
                self.step_count += 1
        
        # Set the new phase
        self.current_phase = action
        self._set_phase(self.current_phase)
        
        # Execute green phase for ACTION_DURATION seconds
        total_reward = 0
        for _ in range(config.ACTION_DURATION):
            if self.conn.simulation.getMinExpectedNumber() <= 0:
                break
            self.conn.simulationStep()
            self.step_count += 1
            total_reward += self._calculate_reward()
        
        # Average reward over the action duration
        reward = total_reward / config.ACTION_DURATION
        
        # Collect metrics at each decision step
        current_queue = self.get_average_queue_length()
        current_wait = self.get_average_waiting_time()
        self.episode_queue_lengths.append(current_queue)
        self.episode_waiting_times.append(current_wait)
        
        # Get next state
        next_state = self._get_state()
        
        # Check if simulation is done
        done = (self.conn.simulation.getMinExpectedNumber() <= 0) or (self.step_count >= config.MAX_STEPS_PER_EPISODE)
        
        info = {
            'step_count': self.step_count,
            'phase': self.current_phase,
            'queue_length': current_queue,
            'waiting_time': current_wait,
        }
        
        return next_state, reward, done, info
    
    def get_episode_avg_queue_length(self):
        """Get the average queue length for the entire episode."""
        if not self.episode_queue_lengths:
            return 0.0
        return np.mean(self.episode_queue_lengths)
    
    def get_episode_avg_waiting_time(self):
        """Get the average waiting time for the entire episode."""
        if not self.episode_waiting_times:
            return 0.0
        return np.mean(self.episode_waiting_times)
    
    def close(self):
        """Close the environment."""
        try:
            self.conn.close()
        except:
            pass
    
    def get_state_size(self):
        """Return the size of the state space."""
        if self.state_size is None:
            raise RuntimeError("Environment must be reset before getting state size")
        return self.state_size
    
    def get_action_size(self):
        """Return the size of the action space."""
        return config.NUM_ACTIONS
