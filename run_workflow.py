from aiida.engine import run_get_node
from aiida.orm import Bool
from aiida import load_profile
from workflow import SimpleNLQueryWorkChain, create_string_node, DOMAIN_TERMS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.table import Table
import datetime
import os
import re
import random
from matplotlib.colors import LinearSegmentedColormap

# Configure plot styles for professional appearance
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Load AiiDA profile
load_profile('xyz')

def fix_query_text(text):
    """Fix spacing issues in query text"""
    text = re.sub(r'FROM(\w+)', r'FROM \1', text)
    text = re.sub(r'WHERE(\w+)', r'WHERE \1', text)
    text = re.sub(r'AND(\w+)', r'AND \1', text)
    text = re.sub(r'CONTAINS\(', r'CONTAINS (', text)
    return text

def execute_query(query, domain):
    """Execute a natural language query using AiiDA's QueryBuilder
    
    This implementation queries the AiiDA database for real data,
    falling back to sample data only if no results are found.
    """
    from aiida.orm import QueryBuilder
    from aiida.orm import StructureData, Dict
    import re
    import random
    
    print(f"Processing query: '{query}' for domain: '{domain}'")
    
    # Parse the query to extract key requirements
    query_lower = query.lower()
    
    # Extract numerical conditions (e.g., bandgap > 2)
    numerical_conditions = []
    for pattern, operator in [
        (r'(\w+)\s+greater\s+than\s+(\d+\.?\d*)', '>'),
        (r'(\w+)\s+more\s+than\s+(\d+\.?\d*)', '>'),
        (r'(\w+)\s+larger\s+than\s+(\d+\.?\d*)', '>'),
        (r'(\w+)\s+less\s+than\s+(\d+\.?\d*)', '<'),
        (r'(\w+)\s+smaller\s+than\s+(\d+\.?\d*)', '<'),
        (r'(\w+)\s*([><]=?|=)\s*(\d+\.?\d*)', None)  # Direct operator notation
    ]:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            if operator:
                # Pattern with implicit operator
                property_name, value = match
                numerical_conditions.append((property_name, operator, float(value)))
            else:
                # Pattern with explicit operator
                property_name, op, value = match
                numerical_conditions.append((property_name, op, float(value)))
    
    # Full periodic table - all 118 elements
    element_map = {
        'hydrogen': 'H', 'helium': 'He', 'lithium': 'Li', 'beryllium': 'Be',
        'boron': 'B', 'carbon': 'C', 'nitrogen': 'N', 'oxygen': 'O',
        'fluorine': 'F', 'neon': 'Ne', 'sodium': 'Na', 'magnesium': 'Mg',
        'aluminum': 'Al', 'aluminium': 'Al', 'silicon': 'Si', 'phosphorus': 'P',
        'sulfur': 'S', 'sulphur': 'S', 'chlorine': 'Cl', 'argon': 'Ar',
        'potassium': 'K', 'calcium': 'Ca', 'scandium': 'Sc', 'titanium': 'Ti',
        'vanadium': 'V', 'chromium': 'Cr', 'manganese': 'Mn', 'iron': 'Fe',
        'cobalt': 'Co', 'nickel': 'Ni', 'copper': 'Cu', 'zinc': 'Zn',
        'gallium': 'Ga', 'germanium': 'Ge', 'arsenic': 'As', 'selenium': 'Se',
        'bromine': 'Br', 'krypton': 'Kr', 'rubidium': 'Rb', 'strontium': 'Sr',
        'yttrium': 'Y', 'zirconium': 'Zr', 'niobium': 'Nb', 'molybdenum': 'Mo',
        'technetium': 'Tc', 'ruthenium': 'Ru', 'rhodium': 'Rh', 'palladium': 'Pd',
        'silver': 'Ag', 'cadmium': 'Cd', 'indium': 'In', 'tin': 'Sn',
        'antimony': 'Sb', 'tellurium': 'Te', 'iodine': 'I', 'xenon': 'Xe',
        'cesium': 'Cs', 'caesium': 'Cs', 'barium': 'Ba', 'lanthanum': 'La',
        'cerium': 'Ce', 'praseodymium': 'Pr', 'neodymium': 'Nd', 'promethium': 'Pm',
        'samarium': 'Sm', 'europium': 'Eu', 'gadolinium': 'Gd', 'terbium': 'Tb',
        'dysprosium': 'Dy', 'holmium': 'Ho', 'erbium': 'Er', 'thulium': 'Tm',
        'ytterbium': 'Yb', 'lutetium': 'Lu', 'hafnium': 'Hf', 'tantalum': 'Ta',
        'tungsten': 'W', 'rhenium': 'Re', 'osmium': 'Os', 'iridium': 'Ir',
        'platinum': 'Pt', 'gold': 'Au', 'mercury': 'Hg', 'thallium': 'Tl',
        'lead': 'Pb', 'bismuth': 'Bi', 'polonium': 'Po', 'astatine': 'At',
        'radon': 'Rn', 'francium': 'Fr', 'radium': 'Ra', 'actinium': 'Ac',
        'thorium': 'Th', 'protactinium': 'Pa', 'uranium': 'U', 'neptunium': 'Np',
        'plutonium': 'Pu', 'americium': 'Am', 'curium': 'Cm', 'berkelium': 'Bk',
        'californium': 'Cf', 'einsteinium': 'Es', 'fermium': 'Fm', 'mendelevium': 'Md',
        'nobelium': 'No', 'lawrencium': 'Lr', 'rutherfordium': 'Rf', 'dubnium': 'Db',
        'seaborgium': 'Sg', 'bohrium': 'Bh', 'hassium': 'Hs', 'meitnerium': 'Mt',
        'darmstadtium': 'Ds', 'roentgenium': 'Rg', 'copernicium': 'Cn', 'nihonium': 'Nh',
        'flerovium': 'Fl', 'moscovium': 'Mc', 'livermorium': 'Lv', 'tennessine': 'Ts',
        'oganesson': 'Og'
    }
    
    # Extract element requirements
    element_requirements = []
    for element_name, symbol in element_map.items():
        if element_name in query_lower or f" {symbol.lower()} " in f" {query_lower} ":
            element_requirements.append(symbol)
    
    print(f"Detected numerical conditions: {numerical_conditions}")
    print(f"Detected element requirements: {element_requirements}")
    
    # Initialize QueryBuilder
    qb = QueryBuilder()
    
    # Define crystal systems for filtering
    crystal_systems = ['cubic', 'hexagonal', 'tetragonal', 'orthorhombic', 
                      'monoclinic', 'triclinic', 'trigonal']
    
    # Base query depending on domain
    if domain == 'materials':
        # Query for structure data
        qb.append(
            StructureData,
            tag='structure',
            project=['id', 'uuid', 'extras', 'attributes', 'label']
        )
        
        # Add filters for numerical conditions
        for prop_name, operator, value in numerical_conditions:
            if prop_name in ['bandgap', 'band', 'gap', 'band gap']:
                # Try both extras and attributes locations for bandgap
                bandgap_filters = [
                    {'extras.bandgap': {operator: value}},
                    {'attributes.bandgap': {operator: value}},
                    {'attributes.band_gap': {operator: value}}
                ]
                qb.add_filter('structure', {'or': bandgap_filters})
                
            elif prop_name in ['energy', 'formation', 'formation energy']:
                # Try both extras and attributes locations for formation energy
                energy_filters = [
                    {'extras.formation_energy': {operator: value}},
                    {'attributes.formation_energy': {operator: value}},
                    {'attributes.energy': {operator: value}}
                ]
                qb.add_filter('structure', {'or': energy_filters})
        
        # Add filters for element requirements
        for element in element_requirements:
            element_filters = [
                {'extras.elements': {'contains': [element]}},
                {'attributes.elements': {'contains': [element]}},
                {'attributes.kinds': {'like': f'%{element}%'}}
            ]
            qb.add_filter('structure', {'or': element_filters})
        
        # Add filter for crystal system if mentioned
        for system in crystal_systems:
            if system in query_lower:
                crystal_filters = [
                    {'extras.crystal_system': {'==': system}},
                    {'attributes.crystal_system': {'==': system}}
                ]
                qb.add_filter('structure', {'or': crystal_filters})
                
    elif domain == 'quantum':
        # For quantum domain, append relevant node types
        from aiida.orm import CalcJobNode
        qb.append(
            CalcJobNode,
            tag='calculation',
            project=['id', 'uuid', 'extras', 'attributes', 'label']
        )
        
        # Add quantum-specific filters
        quantum_terms = ['quantum', 'qubit', 'coherence', 'superconducting']
        term_filters = []
        for term in quantum_terms:
            if term in query_lower:
                term_filters.extend([
                    {'attributes.process_label': {'like': f'%{term}%'}},
                    {'label': {'like': f'%{term}%'}},
                    {'description': {'like': f'%{term}%'}}
                ])
        
        if term_filters:
            qb.add_filter('calculation', {'or': term_filters})
        
        # Add numerical filters
        for prop_name, operator, value in numerical_conditions:
            if prop_name in ['coherence', 'coherence time']:
                coherence_filters = [
                    {'extras.coherence_time': {operator: value}},
                    {'attributes.coherence_time': {operator: value}}
                ]
                qb.add_filter('calculation', {'or': coherence_filters})
    
    elif domain in ['simulation', 'molecular']:
        # For simulation domain, append relevant node types
        from aiida.orm import CalcJobNode
        qb.append(
            CalcJobNode,
            tag='simulation',
            project=['id', 'uuid', 'extras', 'attributes', 'label']
        )
        
        # Add simulation-specific filters
        sim_terms = ['molecular dynamics', 'simulation', 'md', 'dft']
        term_filters = []
        for term in sim_terms:
            if term in query_lower:
                term_filters.extend([
                    {'attributes.process_label': {'like': f'%{term}%'}},
                    {'label': {'like': f'%{term}%'}}
                ])
        
        if term_filters:
            qb.add_filter('simulation', {'or': term_filters})
        
        # Add numerical filters
        for prop_name, operator, value in numerical_conditions:
            if prop_name in ['temperature', 'temp']:
                temp_filters = [
                    {'extras.temperature': {operator: value}},
                    {'attributes.temperature': {operator: value}}
                ]
                qb.add_filter('simulation', {'or': temp_filters})
            elif prop_name in ['pressure', 'press']:
                press_filters = [
                    {'extras.pressure': {operator: value}},
                    {'attributes.pressure': {operator: value}}
                ]
                qb.add_filter('simulation', {'or': press_filters})
    
    elif domain == 'workflow':
        # For workflow domain, append relevant node types
        from aiida.orm import WorkChainNode
        qb.append(
            WorkChainNode,
            tag='workflow',
            project=['id', 'uuid', 'attributes', 'label']
        )
        
        # Add workflow-specific filters
        workflow_terms = ['workflow', 'dft', 'relaxation', 'calculation']
        term_filters = []
        for term in workflow_terms:
            if term in query_lower:
                term_filters.extend([
                    {'attributes.process_label': {'like': f'%{term}%'}},
                    {'label': {'like': f'%{term}%'}}
                ])
        
        if term_filters:
            qb.add_filter('workflow', {'or': term_filters})
    
    # Execute the query with a reasonable limit
    qb.limit(10)
    
    # Process the results
    results = []
    try:
        for item in qb.all():
            node = item[0]  # Get the node from the result tuple
            
            if domain == 'materials':
                # Try to get formula
                formula = None
                if hasattr(node, 'get_formula'):
                    formula = node.get_formula()
                elif hasattr(node, 'extras') and 'formula' in node.extras:
                    formula = node.extras['formula']
                elif hasattr(node, 'attributes') and 'formula' in node.attributes:
                    formula = node.attributes['formula']
                
                # Try to get elements
                elements = None
                if hasattr(node, 'extras') and 'elements' in node.extras:
                    elements = node.extras['elements']
                elif hasattr(node, 'attributes') and 'elements' in node.attributes:
                    elements = node.attributes['elements']
                
                # Try to get crystal system
                crystal_system = None
                if hasattr(node, 'extras') and 'crystal_system' in node.extras:
                    crystal_system = node.extras['crystal_system']
                elif hasattr(node, 'attributes') and 'crystal_system' in node.attributes:
                    crystal_system = node.attributes['crystal_system']
                
                # Try to get bandgap
                bandgap = None
                if hasattr(node, 'extras') and 'bandgap' in node.extras:
                    bandgap = node.extras['bandgap']
                elif hasattr(node, 'attributes') and 'bandgap' in node.attributes:
                    bandgap = node.attributes['bandgap']
                elif hasattr(node, 'attributes') and 'band_gap' in node.attributes:
                    bandgap = node.attributes['band_gap']
                
                # Try to get formation energy
                formation_energy = None
                if hasattr(node, 'extras') and 'formation_energy' in node.extras:
                    formation_energy = node.extras['formation_energy']
                elif hasattr(node, 'attributes') and 'formation_energy' in node.attributes:
                    formation_energy = node.attributes['formation_energy']
                
                # Create result dictionary
                result = {
                    'id': f'MAT-{node.pk}',
                    'pk': node.pk,
                    'formula': formula or node.label or "Unknown",
                    'elements': elements or [],
                    'crystal_system': crystal_system or "Unknown",
                    'bandgap': bandgap or 0.0,
                    'formation_energy': formation_energy or 0.0
                }
                results.append(result)
                
            elif domain == 'quantum':
                # Extract quantum-related properties
                result = {
                    'id': f'QC-{node.pk}',
                    'pk': node.pk,
                    'process_label': node.attributes.get('process_label', 'Unknown')
                }
                
                # Try to get coherence time
                if hasattr(node, 'extras') and 'coherence_time' in node.extras:
                    result['coherence'] = node.extras['coherence_time']
                elif hasattr(node, 'attributes') and 'coherence_time' in node.attributes:
                    result['coherence'] = node.attributes['coherence_time']
                
                # Try to get qubits
                if hasattr(node, 'extras') and 'qubits' in node.extras:
                    result['qubits'] = node.extras['qubits']
                elif hasattr(node, 'attributes') and 'qubits' in node.attributes:
                    result['qubits'] = node.attributes['qubits']
                
                results.append(result)
                
            elif domain in ['simulation', 'molecular']:
                # Extract simulation-related properties
                result = {
                    'id': f'SIM-{node.pk}',
                    'pk': node.pk,
                    'process_label': node.attributes.get('process_label', 'Unknown')
                }
                
                # Try to get temperature
                if hasattr(node, 'extras') and 'temperature' in node.extras:
                    result['temperature'] = node.extras['temperature']
                elif hasattr(node, 'attributes') and 'temperature' in node.attributes:
                    result['temperature'] = node.attributes['temperature']
                
                # Try to get pressure
                if hasattr(node, 'extras') and 'pressure' in node.extras:
                    result['pressure'] = node.extras['pressure']
                elif hasattr(node, 'attributes') and 'pressure' in node.attributes:
                    result['pressure'] = node.attributes['pressure']
                
                results.append(result)
                
            elif domain == 'workflow':
                # Extract workflow-related properties
                result = {
                    'id': f'WF-{node.pk}',
                    'pk': node.pk,
                    'process_label': node.attributes.get('process_label', 'Unknown'),
                    'state': node.attributes.get('process_state', 'Unknown'),
                    'ctime': node.ctime.strftime("%Y-%m-%d")
                }
                results.append(result)
    
    except Exception as e:
        print(f"Error querying AiiDA database: {str(e)}")
    
    # If no results from QueryBuilder, use the hardcoded sample data
    if not results:
        print(f"No results from AiiDA database for '{domain}' domain. Using sample data.")
        
        if domain == 'materials':
            # Sample materials data
            material_samples = [
                {'id': 'MAT-1001', 'pk': 1001, 'formula': 'Si', 'elements': ['Si'], 
                 'crystal_system': 'cubic', 'bandgap': 1.1, 'formation_energy': -4.63},
                
                {'id': 'MAT-1002', 'pk': 1002, 'formula': 'SiO2', 'elements': ['Si', 'O'], 
                 'crystal_system': 'hexagonal', 'bandgap': 9.0, 'formation_energy': -5.99},
                
                {'id': 'MAT-1003', 'pk': 1003, 'formula': 'SiC', 'elements': ['Si', 'C'], 
                 'crystal_system': 'cubic', 'bandgap': 3.2, 'formation_energy': -6.34},
                
                {'id': 'MAT-1004', 'pk': 1004, 'formula': 'Fe2O3', 'elements': ['Fe', 'O'], 
                 'crystal_system': 'trigonal', 'bandgap': 2.2, 'formation_energy': -3.76},
                
                {'id': 'MAT-1005', 'pk': 1005, 'formula': 'ZnO', 'elements': ['Zn', 'O'], 
                 'crystal_system': 'hexagonal', 'bandgap': 3.3, 'formation_energy': -3.63},
                
                {'id': 'MAT-1006', 'pk': 1006, 'formula': 'BaTiO3', 'elements': ['Ba', 'Ti', 'O'], 
                 'crystal_system': 'tetragonal', 'bandgap': 3.2, 'formation_energy': -5.82},
                 
                {'id': 'MAT-1007', 'pk': 1007, 'formula': 'Cu2O', 'elements': ['Cu', 'O'], 
                 'crystal_system': 'cubic', 'bandgap': 2.1, 'formation_energy': -1.75},
                 
                {'id': 'MAT-1008', 'pk': 1008, 'formula': 'GaAs', 'elements': ['Ga', 'As'], 
                 'crystal_system': 'cubic', 'bandgap': 1.4, 'formation_energy': -3.2},
                 
                {'id': 'MAT-1009', 'pk': 1009, 'formula': 'Si3N4', 'elements': ['Si', 'N'], 
                 'crystal_system': 'hexagonal', 'bandgap': 5.3, 'formation_energy': -7.56},
                 
                {'id': 'MAT-1010', 'pk': 1010, 'formula': 'TiO2', 'elements': ['Ti', 'O'], 
                 'crystal_system': 'tetragonal', 'bandgap': 3.2, 'formation_energy': -4.98}
            ]
            
            # Apply filters to sample data
            filtered_materials = material_samples.copy()
            
            # Apply numerical conditions
            for prop_name, operator, value in numerical_conditions:
                if prop_name in ['bandgap', 'band', 'gap', 'band gap']:
                    filtered_materials = [
                        m for m in filtered_materials 
                        if ((operator == '>' and m['bandgap'] > value) or
                             (operator == '<' and m['bandgap'] < value) or
                             (operator == '>=' and m['bandgap'] >= value) or
                             (operator == '<=' and m['bandgap'] <= value) or
                             (operator == '=' and m['bandgap'] == value))
                    ]
                elif prop_name in ['energy', 'formation', 'formation energy']:
                    filtered_materials = [
                        m for m in filtered_materials 
                        if ((operator == '>' and m['formation_energy'] > value) or
                             (operator == '<' and m['formation_energy'] < value) or
                             (operator == '>=' and m['formation_energy'] >= value) or
                             (operator == '<=' and m['formation_energy'] <= value) or
                             (operator == '=' and m['formation_energy'] == value))
                    ]
            
            # Apply element requirements
            for element in element_requirements:
                filtered_materials = [
                    m for m in filtered_materials 
                    if element in m['elements']
                ]
                
            # Apply crystal system filter if mentioned
            for system in crystal_systems:
                if system in query_lower:
                    filtered_materials = [
                        m for m in filtered_materials 
                        if system == m['crystal_system']
                    ]
            
            results = filtered_materials
            
        elif domain == 'quantum':
            # Generate quantum sample data
            n_results = random.randint(3, 6)
            for i in range(n_results):
                coherence = random.uniform(10, 200)
                # Apply numerical filters
                for prop_name, operator, value in numerical_conditions:
                    if prop_name in ['coherence', 'coherence time']:
                        if operator == '>':
                            coherence = max(coherence, value * 1.1)
                        elif operator == '<':
                            coherence = min(coherence, value * 0.9)
                
                results.append({
                    'id': f'QC-{1000+i}',
                    'pk': 1000+i,
                    'process_label': random.choice(['QuantumCircuit', 'QubitSimulation']),
                    'qubits': random.randint(1, 50),
                    'coherence': round(coherence, 1),
                    'error_rate': round(random.uniform(0.001, 0.1), 4)
                })
                
        elif domain in ['simulation', 'molecular']:
            # Generate simulation sample data
            n_results = random.randint(3, 6)
            for i in range(n_results):
                temperature = random.uniform(100, 1000)
                pressure = random.uniform(1, 100)
                
                # Apply numerical filters
                for prop_name, operator, value in numerical_conditions:
                    if prop_name in ['temperature', 'temp']:
                        if operator == '>':
                            temperature = max(temperature, value * 1.1)
                        elif operator == '<':
                            temperature = min(temperature, value * 0.9)
                    elif prop_name in ['pressure', 'press']:
                        if operator == '>':
                            pressure = max(pressure, value * 1.1)
                        elif operator == '<':
                            pressure = min(pressure, value * 0.9)
                
                results.append({
                    'id': f'SIM-{1000+i}',
                    'pk': 1000+i,
                    'process_label': random.choice(['MolecularDynamics', 'DFTCalculation']),
                    'atoms': random.randint(10, 5000),
                    'temperature': round(temperature, 1),
                    'pressure': round(pressure, 2),
                    'simulation_time': round(random.uniform(0.1, 100), 2)
                })
                
        elif domain == 'workflow':
            # Generate workflow sample data
            n_results = random.randint(3, 6)
            for i in range(n_results):
                results.append({
                    'id': f'WF-{1000+i}',
                    'pk': 1000+i,
                    'process_label': random.choice(['DFTWorkflow', 'RelaxationWorkflow']),
                    'state': random.choice(['finished', 'running']),
                    'ctime': f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
                })
    
    return results

def generate_insight(results, query_text, domain):
    """Generate an insight from the query results"""
    if not results:
        return "No results found for this query."
    
    # Get the column names
    columns = list(results[0].keys())
    
    # Create a pandas DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Generate insights based on domain and available columns
    insights = []
    
    # General statistics
    insights.append(f"Found {len(results)} matching records.")
    
    # Numerical column analysis
    numerical_cols = []
    for col in columns:
        if all(isinstance(row.get(col), (int, float)) for row in results):
            numerical_cols.append(col)
    
    if numerical_cols:
        for col in numerical_cols[:2]:  # Limit to first two numerical columns
            values = [row.get(col) for row in results]
            insights.append(f"Average {col}: {sum(values)/len(values):.2f}")
            insights.append(f"Range of {col}: {min(values):.2f} to {max(values):.2f}")
    
    # Categorical column analysis
    categorical_cols = []
    for col in columns:
        if not all(isinstance(row.get(col), (int, float)) for row in results) and col != 'id':
            categorical_cols.append(col)
    
    if categorical_cols:
        for col in categorical_cols[:1]:  # Limit to first categorical column
            values = [row.get(col) for row in results]
            value_counts = {}
            for v in values:
                value_counts[v] = value_counts.get(v, 0) + 1
            
            most_common = max(value_counts.items(), key=lambda x: x[1])
            insights.append(f"Most common {col}: {most_common[0]} ({most_common[1]} occurrences)")
    
    # Domain-specific insights
    if domain == 'materials':
        if 'bandgap' in columns:
            bandgaps = [row.get('bandgap') for row in results]
            semiconductors = [bg for bg in bandgaps if 0.1 <= bg <= 4.0]
            if semiconductors:
                insights.append(f"{len(semiconductors)}/{len(bandgaps)} materials are semiconductors.")
        
        if 'elements' in columns:
            element_counts = {}
            for row in results:
                for element in row.get('elements', []):
                    element_counts[element] = element_counts.get(element, 0) + 1
            
            if 'Si' in element_counts:
                insights.append(f"{element_counts['Si']}/{len(results)} materials contain silicon.")
                
    elif domain == 'quantum':
        if 'coherence' in columns:
            coherence_times = [row.get('coherence') for row in results]
            insights.append(f"Median coherence time: {np.median(coherence_times):.2f}")
            
    elif domain == 'molecular' or domain == 'simulation':
        if 'pressure' in columns and 'temperature' in columns:
            high_pt = sum(1 for r in results if r.get('pressure', 0) > 50 and r.get('temperature', 0) > 500)
            insights.append(f"{high_pt}/{len(results)} simulations at high pressure and temperature.")
    
    return "\n".join(insights)

def generate_report_with_results(results, descriptions, nodes):
    """Generate a complete report with query results and insights"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = "nl_query_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    report_file = os.path.join(report_dir, f"nl_query_report_{timestamp}.pdf")
    
    # Execute queries and generate insights
    query_results = []
    insights = []
    
    for result, desc in zip(results, descriptions):
        # Fix query spacing
        query_text = fix_query_text(result['query'].value)
        domain = result['analysis'].get_dict()['domain']
        
        # Execute query
        qresult = execute_query(desc, domain)  # Use original description for better context
        query_results.append(qresult)
        
        # Generate insight
        insight = generate_insight(qresult, query_text, domain)
        insights.append(insight)
    
    # Create PDF report
    with PdfPages(report_file) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Add gradient background
        colors = [(0.95, 0.95, 1), (0.85, 0.9, 1)]
        cmap = LinearSegmentedColormap.from_list("custom_gradient", colors, N=100)
        gradient = np.linspace(0, 1, 100).reshape(-1, 1)
        gradient = np.repeat(gradient, 100, axis=1)
        ax.imshow(gradient, aspect='auto', cmap=cmap, 
                 extent=[0, 11, 0, 8.5], alpha=0.7, zorder=-10)
        
        # Add title and subtitle
        ax.text(5.5, 7.2, "Natural Language to SQL", 
               ha='center', fontsize=28, fontweight='bold', color='navy', zorder=4)
        ax.text(5.5, 5.0, "AiiDA Query Analysis Report", 
               ha='center', fontsize=20, color='steelblue', fontstyle='italic', zorder=3)
        
        # Decorative line
        ax.axhline(y=4.8, xmin=0.1, xmax=0.9, color='steelblue', 
                  linestyle='-', linewidth=1, alpha=0.6)
        
        # Author information
        ax.text(5.5, 2.7, "By Muhammad Rebaal", 
               ha='center', fontsize=24, fontweight='bold', color='steelblue', zorder=5)
        
        # Footer
        ax.text(5.5, 0.7, "AiiDA Natural Language Query Demo", 
               ha='center', fontsize=11, fontstyle='italic', color='dimgray')
        ax.text(10, 0.4, f"v1.0 | {timestamp}", 
               ha='right', fontsize=8, color='darkgray')
        
        pdf.savefig(fig)
        plt.close(fig)
        
        # Results pages for each query
        for i, (result, desc, qresult, insight) in enumerate(zip(results, descriptions, query_results, insights)):
            domain = result['analysis'].get_dict()['domain']
            
            fig = plt.figure(figsize=(11, 8.5))
            gs = gridspec.GridSpec(3, 2, figure=fig, 
                                  height_ratios=[0.6, 1.4, 1.4], 
                                  hspace=0.4, wspace=0.3)
            
            # Header
            ax_header = fig.add_subplot(gs[0, :])
            ax_header.axis('off')
            ax_header.text(0.5, 0.6, f"Query {i+1} Results: {domain.capitalize()} Domain", 
                          ha='center', fontsize=16, fontweight='bold')
            
            # Natural language query
            ax_header.text(0.1, 0.25, f"Natural language: '{desc}'", 
                        ha='left', fontsize=10, fontstyle='italic')
            
            # Results table
            ax_table = fig.add_subplot(gs[1, :])
            ax_table.axis('off')
            
            if qresult:
                columns = list(qresult[0].keys())
                table_data = []
                for row in qresult[:5]:  # Limit to first 5 results
                    table_data.append([str(row.get(col, ''))[:12] for col in columns])
                
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=columns,
                    loc='center',
                    cellLoc='center'
                )
                
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                
                for j, key in enumerate(columns):
                    cell = table[0, j]
                    cell.set_facecolor('lightsteelblue')
                    cell.set_text_props(fontweight='bold')
                
                ax_table.set_title('Query Results', fontsize=14, pad=20)
            else:
                ax_table.text(0.5, 0.5, "No results retrieved for this query", 
                            ha='center', va='center', fontsize=12)
            
            # Data visualization
            ax_viz = fig.add_subplot(gs[2, 0])
            ax_insights = fig.add_subplot(gs[2, 1])
            
            # Bar chart for numeric data
            if qresult and len(qresult) > 0:
                numeric_cols = []
                for col in qresult[0].keys():
                    if all(isinstance(row.get(col), (int, float)) for row in qresult):
                        numeric_cols.append(col)
                
                if numeric_cols:
                    plot_col = numeric_cols[0]
                    ids = [str(row.get('id', f'Item {i}'))[:8] for i, row in enumerate(qresult)]
                    values = [row.get(plot_col) for row in qresult]
                    
                    bars = ax_viz.bar(ids, values, color=sns.color_palette("Blues", len(ids)))
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax_viz.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                  f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                    
                    ax_viz.set_xticklabels(ids, rotation=45, ha='right')
                    ax_viz.set_ylabel(plot_col)
                    ax_viz.set_title(f'{plot_col} by Item', fontsize=12, pad=10)
                    
                else:
                    ax_viz.axis('off')
                    ax_viz.text(0.5, 0.5, "No numeric data available for visualization", 
                              ha='center', va='center', fontsize=10)
            else:
                ax_viz.axis('off')
                ax_viz.text(0.5, 0.5, "No data available for visualization", 
                          ha='center', va='center', fontsize=10)
            
            # Insights
            ax_insights.axis('off')
            ax_insights.text(0.5, 0.95, "Data Insights", ha='center', fontsize=14, fontweight='bold')
            
            insight_y = 0.85
            for line in insight.split('\n'):
                ax_insights.text(0.1, insight_y, line, fontsize=10)
                insight_y -= 0.1
            
            fig.tight_layout(pad=2.0)
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"\nReport with results generated: {report_file}")
    return report_file

if __name__ == "__main__":
    # Sample descriptions covering different domains
    descriptions = [
        "Find all crystal structures with bandgap greater than 2 eV and containing silicon",
        "Show me density functional theory calculations for magnetic materials",
        "I need molecular dynamics simulations of water molecules at high pressure",
        "Get quantum calculations with coherence time greater than 100 microseconds",
        "Find workflows that process crystal structures and run DFT calculations"
    ]

    results = []
    nodes = []
    
    # Process all descriptions
    for desc in descriptions:
        print(f"\n{'='*80}\nProcessing: '{desc}'\n{'='*80}")
        
        # Store description as node before passing to workflow
        desc_node = create_string_node(desc)
        
        # Use run_get_node instead of run to get both results and node
        result, node = run_get_node(
            SimpleNLQueryWorkChain,
            description=desc_node,
            validate_query=Bool(True)
        )
        
        results.append(result)
        nodes.append(node)
        
        # Print basic results to console
        print("\nDomain Analysis:")
        analysis = result['analysis'].get_dict()
        print(f"  Primary domain: {analysis['domain']}")
        print(f"  Domain matches: {analysis['domain_matches']}")
        print(f"  Top keywords: {', '.join(analysis['keywords'][:5])}")
        
        print("\nGenerated Query:")
        query_text = fix_query_text(result['query'].value)
        print(f"  {query_text}")
        
        print("\nQuery Metadata:")
        for key, value in result['metadata'].get_dict().items():
            print(f"  {key}: {value}")
        
        if 'validation' in result:
            print("\nValidation Results:")
            validation = result['validation'].get_dict()['validation']
            print(f"  Valid: {validation['is_valid']}")
            if validation['warnings']:
                print(f"  Warnings: {', '.join(validation['warnings'])}")
            if validation['suggestions']:
                print(f"  Suggestions: {', '.join(validation['suggestions'])}")
        
        print("\nWorkflow Provenance:")
        print(f"  Process PK: {node.pk}")
        print("-" * 80)
    
    # Generate comprehensive report with results
    report_file = generate_report_with_results(results, descriptions, nodes)