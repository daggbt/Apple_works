import numpy as np
from collections import defaultdict
import heapq
from typing import Dict, List, Tuple, Set

class Town:
    def __init__(self, id: str, base_demand: float, growth_rate: float):
        self.id = id
        self.base_demand = base_demand
        self.growth_rate = growth_rate
        self.current_demand = base_demand
        self.connected_orchards = []
        
    def update_demand(self, month: int):
        """Update demand based on growth rate and month"""
        self.current_demand = self.base_demand * (1 + self.growth_rate) ** month

class Orchard:
    def __init__(self, id: str, capacity: float, depletion_rate: float):
        self.id = id
        self.max_capacity = capacity
        self.current_capacity = capacity
        self.depletion_rate = depletion_rate
        self.is_regenerating = False
        self.connected_towns = []
        self.regeneration_start = -1  # Initialize with invalid month
        self.regeneration_duration = 0  # Initialize with zero duration
    
    def update_capacity(self, load: float):
        """Update capacity based on usage and depletion"""
        if not self.is_regenerating:
            depletion = load * self.depletion_rate
            self.current_capacity = max(0, self.current_capacity - depletion)
    
    def start_regeneration(self, start_month: int, duration: int):
        """Start regeneration process"""
        self.is_regenerating = True
        self.regeneration_start = start_month
        self.regeneration_duration = duration
    
    def check_regeneration_status(self, current_month: int):
        """Check if regeneration is complete"""
        if self.is_regenerating and self.regeneration_start >= 0:
            if current_month >= self.regeneration_start + self.regeneration_duration:
                self.is_regenerating = False
                self.current_capacity = self.max_capacity
                self.regeneration_start = -1
                self.regeneration_duration = 0

class AppleNetwork:
    def __init__(self):
        self.towns: Dict[str, Town] = {}
        self.orchards: Dict[str, Orchard] = {}
        self.routes: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        
    def add_town(self, id: str, base_demand: float, growth_rate: float):
        """Add a town to the network"""
        self.towns[id] = Town(id, base_demand, growth_rate)
    
    def add_orchard(self, id: str, capacity: float, depletion_rate: float, 
                   town1_id: str, town2_id: str):
        """Add an orchard connection between two towns"""
        self.orchards[id] = Orchard(id, capacity, depletion_rate)
        self.orchards[id].connected_towns = [town1_id, town2_id]
        self.towns[town1_id].connected_orchards.append(id)
        self.towns[town2_id].connected_orchards.append(id)
        
    def find_all_routes(self, start: str, end: str, excluded_orchards: Set[str] = None) -> List[List[str]]:
        """Find all possible routes between two towns"""
        if excluded_orchards is None:
            excluded_orchards = set()
            
        def dfs(current: str, target: str, path: List[str], visited: Set[str], routes: List[List[str]]):
            if current == target:
                routes.append(path[:])
                return
            
            for orchard_id in self.towns[current].connected_orchards:
                if orchard_id not in excluded_orchards and orchard_id not in visited:
                    orchard = self.orchards[orchard_id]
                    next_town = (orchard.connected_towns[1] 
                               if orchard.connected_towns[0] == current 
                               else orchard.connected_towns[0])
                    
                    if next_town not in visited:
                        visited.add(orchard_id)
                        dfs(next_town, target, path + [orchard_id], visited, routes)
                        visited.remove(orchard_id)
        
        routes = []
        dfs(start, end, [], set(), routes)
        return routes

    def optimize_regeneration_schedule(self, simulation_months: int = 12) -> Dict:
      """Enhanced regeneration schedule optimization"""
      schedule = {}
      orchard_priority = []
      network_state = self.analyze_network_state()
      
      for orchard_id, orchard in self.orchards.items():
          # Enhanced priority scoring
          capacity_score = 1 - orchard.current_capacity/orchard.max_capacity
          connectivity_score = len(orchard.connected_towns) / len(self.towns)
          demand_pressure = sum(
              self.towns[tid].current_demand 
              for tid in orchard.connected_towns
          ) / orchard.max_capacity
          
          priority_score = (
              0.4 * capacity_score + 
              0.3 * connectivity_score + 
              0.3 * demand_pressure
          )
          heapq.heappush(orchard_priority, (-priority_score, orchard_id))
      
      regenerating_periods = defaultdict(set)
      for month in range(simulation_months):
          candidates = []
          while orchard_priority and len(candidates) < 3:  # Consider top 3 candidates
              score, oid = heapq.heappop(orchard_priority)
              if self.check_regeneration_feasibility(oid, month):
                  candidates.append((score, oid))
          
          if candidates:
              # Select best candidate based on network impact
              best_candidate = min(
                  candidates,
                  key=lambda x: self.simulate_regeneration_impact(x[1], month)
              )
              _, selected_orchard = best_candidate
              
              duration = self.calculate_regeneration_duration(
                  self.orchards[selected_orchard]
              )
              schedule[selected_orchard] = {
                  'start_month': month,
                  'duration': duration
              }
              
              # Mark regeneration period
              for m in range(month, month + duration):
                  regenerating_periods[m].add(selected_orchard)
          
          # Return unselected candidates to queue
          for score, oid in candidates:
              if oid != selected_orchard:
                  heapq.heappush(orchard_priority, (score, oid))
      
      return schedule

    def analyze_network_state(self) -> Dict:
      """Analyze current network state for optimization"""
      return {
          'total_demand': sum(t.current_demand for t in self.towns.values()),
          'total_capacity': sum(o.current_capacity for o in self.orchards.values()),
          'average_connectivity': sum(
              len(o.connected_towns) for o in self.orchards.values()
          ) / len(self.orchards),
          'critical_paths': self.identify_critical_paths()
      }

    def check_regeneration_feasibility(self, orchard_id: str, month: int) -> bool:
      """Enhanced feasibility check for regeneration"""
      orchard = self.orchards[orchard_id]
      
      # Check minimum throughput maintenance
      for town_id in orchard.connected_towns:
          town = self.towns[town_id]
          
          # Calculate available capacity without this orchard
          available_capacity = sum(
              self.orchards[o_id].current_capacity
              for o_id in town.connected_orchards
              if o_id != orchard_id and not self.orchards[o_id].is_regenerating
          )
          
          # Check if minimum throughput can be maintained
          if available_capacity < town.current_demand * 0.7:
              return False
      
      return True

    def get_adjacent_orchards(self, orchard_id: str) -> Set[str]:
      """Get orchards that share a town with the given orchard"""
      orchard = self.orchards[orchard_id]
      adjacent = set()
      
      for town_id in orchard.connected_towns:
          adjacent.update(self.towns[town_id].connected_orchards)
      
      adjacent.remove(orchard_id)
      return adjacent

    def identify_critical_paths(self) -> List[str]:
      """
      Identify critical paths in the network.
      Critical paths are those essential for maintaining minimum throughput.
      """
      critical_paths = set()
      
      for town_id, town in self.towns.items():
          # Find minimum required capacity for the town
          min_required_capacity = town.current_demand * 0.7
          
          # Check each orchard connected to this town
          connected_orchards = town.connected_orchards
          total_capacity = sum(
              self.orchards[oid].current_capacity 
              for oid in connected_orchards
          )
          
          # If total capacity is close to minimum required, all paths are critical
          if total_capacity < min_required_capacity * 1.2:  # 20% buffer
              critical_paths.update(connected_orchards)
              continue
          
          # Check each orchard's contribution
          for orchard_id in connected_orchards:
              capacity_without_orchard = sum(
                  self.orchards[oid].current_capacity 
                  for oid in connected_orchards 
                  if oid != orchard_id
              )
              
              # If removing this orchard would break minimum capacity requirement
              if capacity_without_orchard < min_required_capacity:
                  critical_paths.add(orchard_id)
      
      return list(critical_paths)


    def simulate_regeneration_impact(self, orchard_id: str, month: int) -> float:
      """
      Simulate the impact of regenerating an orchard at a given month.
      Returns an impact score (lower is better).
      """
      impact_score = 0
      orchard = self.orchards[orchard_id]
      
      # Check impact on connected towns
      for town_id in orchard.connected_towns:
          town = self.towns[town_id]
          
          # Calculate current available capacity
          current_capacity = sum(
              self.orchards[oid].current_capacity 
              for oid in town.connected_orchards 
              if not self.orchards[oid].is_regenerating
          )
          
          # Calculate capacity if this orchard is regenerating
          capacity_without_orchard = current_capacity - orchard.current_capacity
          
          # Calculate impact on throughput
          min_required = town.current_demand * 0.7
          if capacity_without_orchard < min_required:
              impact_score += (min_required - capacity_without_orchard) / min_required
          
          # Consider alternative routes
          alternative_routes = len(self.find_all_routes(
              town_id, 
              [t for t in self.towns if t != town_id][0],  # Any other town
              {orchard_id}
          ))
          impact_score -= 0.1 * alternative_routes  # Reduce impact if alternatives exist
      
      return impact_score
    
    
    def optimize_network_flow(self, month: int) -> Dict[str, float]:
      """Optimize network flow using modified Ford-Fulkerson algorithm"""
      flow_distribution = {}
      available_orchards = {
          oid: orchard for oid, orchard in self.orchards.items() 
          if not orchard.is_regenerating
      }
      
      # Sort towns by demand/capacity ratio
      towns_priority = sorted(
          self.towns.items(),
          key=lambda x: x[1].current_demand / sum(
              self.orchards[oid].current_capacity 
              for oid in x[1].connected_orchards 
              if oid in available_orchards
          ),
          reverse=True
      )
      
      # Distribute flow with priority to higher demand/capacity ratio
      for town_id, town in towns_priority:
          required_flow = town.current_demand
          min_required = required_flow * 0.7  # 70% minimum requirement
          
          # Calculate available capacity from connected orchards
          available_capacity = sum(
              orchard.current_capacity / len(orchard.connected_towns)
              for oid, orchard in available_orchards.items()
              if town_id in orchard.connected_towns
          )
          
          # Optimize flow distribution
          allocated_flow = min(
              max(min_required, available_capacity * 0.8),  # Try to allocate more than minimum
              required_flow  # But don't exceed demand
          )
          
          flow_distribution[town_id] = allocated_flow
          
          # Update remaining capacities
          for oid, orchard in available_orchards.items():
              if town_id in orchard.connected_towns:
                  orchard.current_capacity -= (
                      allocated_flow / len(orchard.connected_towns)
                  )
      
      return flow_distribution
    
    def calculate_regeneration_duration(self, orchard: Orchard) -> int:
      """Calculate optimal regeneration duration with adaptive logic"""
      capacity_deficit = orchard.max_capacity - orchard.current_capacity
      base_duration = max(1, int(np.ceil(capacity_deficit / orchard.depletion_rate)))
      
      # Consider connected towns' demands
      total_dependent_demand = sum(
          self.towns[tid].current_demand 
          for tid in orchard.connected_towns
      )
      
      # Adjust duration based on demand pressure
      demand_factor = total_dependent_demand / orchard.max_capacity
      adjusted_duration = base_duration * (1 + 0.2 * demand_factor)  # Up to 20% longer
      
      return max(1, min(3, int(np.ceil(adjusted_duration))))  # Cap at 3 months
    
    def simulate_network(self, schedule: Dict, months: int = 12) -> Dict:
        """Simulate network operation with given schedule"""
        results = defaultdict(dict)
        
        for month in range(months):
            # Update demands
            for town in self.towns.values():
                town.update_demand(month)
            
            # Apply regeneration schedule and check status
            for orchard_id, orchard in self.orchards.items():
                orchard.check_regeneration_status(month)
                
                if orchard_id in schedule:
                    regen_info = schedule[orchard_id]
                    if month == regen_info['start_month']:
                        orchard.start_regeneration(month, regen_info['duration'])
            
            # Calculate throughput and record results
            for town_id, town in self.towns.items():
                throughput = self.calculate_town_throughput(town_id, month)
                results[month][town_id] = {
                    'demand': town.current_demand,
                    'throughput': throughput,
                    'satisfaction_rate': throughput / town.current_demand if town.current_demand > 0 else 0
                }
        
        return results
    
    def calculate_town_throughput(self, town_id: str, month: int) -> float:
      """Calculate actual throughput to a town with proper load balancing"""
      town = self.towns[town_id]
      required_throughput = town.current_demand * 0.7  # Minimum required throughput
      
      # Get all available routes to this town
      available_capacity = 0
      active_orchards = []
      
      for orch_id in town.connected_orchards:
          orchard = self.orchards[orch_id]
          if not orchard.is_regenerating:
              active_orchards.append(orchard)
              # Divide capacity by number of connected towns to ensure fair distribution
              available_capacity += orchard.current_capacity / len(orchard.connected_towns)
      
      # Calculate fair share based on demand proportion
      total_connected_demand = sum(
          self.towns[t_id].current_demand 
          for orch in active_orchards 
          for t_id in orch.connected_towns
      )
      
      if total_connected_demand > 0:
          fair_share = (town.current_demand / total_connected_demand) * available_capacity
          # Ensure minimum throughput requirement
          return max(required_throughput, min(fair_share, town.current_demand))
      
      return 0

def main():
    # Initialize network
    network = AppleNetwork()
    
    # Add towns with more realistic values
    towns = ['T1', 'T2', 'T3', 'T4', 'T5']
    for i, town_id in enumerate(towns):
        network.add_town(town_id, base_demand=500 * (i + 1), growth_rate=0.02)  # 2% monthly growth
    
    # Add orchards with realistic capacities
    orchard_connections = [
        ('O1', 'T1', 'T2', 2500), ('O2', 'T2', 'T3', 3000),
        ('O3', 'T3', 'T4', 2800), ('O4', 'T4', 'T5', 2600),
        ('O5', 'T5', 'T1', 2700), ('O6', 'T1', 'T3', 3200),
        ('O7', 'T2', 'T4', 2900)
    ]
    
    for o_id, t1, t2, cap in orchard_connections:
        network.add_orchard(o_id, capacity=cap, depletion_rate=0.05,  # 5% monthly depletion
                          town1_id=t1, town2_id=t2)
    
    # Generate regeneration schedule
    print("Generating regeneration schedule...")
    schedule = network.optimize_regeneration_schedule()
    
    # Simulate network operation
    print("Simulating network operation...")
    results = network.simulate_network(schedule)
    
    # Print results
    print("\nRegeneration Schedule:")
    for orchard_id, info in schedule.items():
        print(f"Orchard {orchard_id}: Start Month {info['start_month']}, "
              f"Duration {info['duration']} months")
    
    print("\nSimulation Results:")
    for month in range(12):
        print(f"\nMonth {month}:")
        for town_id, metrics in results[month].items():
            print(f"Town {town_id}: Demand={metrics['demand']:.0f}, "
                  f"Throughput={metrics['throughput']:.0f}, "
                  f"Satisfaction={metrics['satisfaction_rate']:.2%}")

if __name__ == "__main__":
    main()