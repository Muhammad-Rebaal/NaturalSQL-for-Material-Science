from aiida.engine import calcfunction, WorkChain, run
from aiida.orm import Dict, Str, Bool
from aiida import load_profile
from collections import Counter, defaultdict
import re
import string

# Load your AiiDA profile
load_profile('xyz')

# Basic stopwords list (most common English words to filter out)
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'when', 'where', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', 'should', 'now', 'with', 'for', 'by', 'from', 'to', 'of', 'in', 'on', 'at', 'up',
    'down', 'that', 'this', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'it', 'its', 'i', 'me',
    'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their', 'theirs'
}

# Scientific domain-specific terms for better classification
DOMAIN_TERMS = {
    'materials': ['crystal', 'structure', 'material', 'atom', 'molecule', 'band', 'gap', 'energy', 
                  'dft', 'density', 'functional', 'magnetic', 'superconductor', 'metal', 'oxide'],
    'simulation': ['molecular', 'dynamics', 'monte', 'carlo', 'simulation', 'model', 'calculation',
                  'force', 'field', 'temperature', 'pressure', 'volume'],
    'quantum': ['quantum', 'qbit', 'qubit', 'entanglement', 'superposition', 'coherence', 
                'decoherence', 'gate', 'circuit', 'algorithm'],
    'workflow': ['workflow', 'process', 'pipeline', 'step', 'calculation', 'protocol', 'procedure']
}

# Query templates based on classification
QUERY_TEMPLATES = {
    'materials': "SELECT * FROM materials WHERE {conditions}",
    'simulation': "SELECT * FROM simulations WHERE {conditions}",
    'quantum': "SELECT * FROM quantum_calculations WHERE {conditions}",
    'workflow': "SELECT * FROM workflows WHERE {conditions}",
    'default': "SELECT * FROM data WHERE {conditions}"
}

def simple_tokenize(text):
    """Simple tokenization without NLTK."""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split by whitespace
    return text.split()

def simple_sentence_split(text):
    """Split text into sentences without NLTK."""
    # Replace common sentence terminators with special marker
    text = re.sub(r'[.!?]', '###', text)
    # Split by the marker
    sentences = text.split('###')
    # Remove empty sentences and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

@calcfunction
def preprocess_text(description):
    """Preprocess text without using NLTK."""
    text = description.value.lower()
    
    # Split into sentences
    sentences = simple_sentence_split(text)
    
    # Tokenize and filter
    tokens = []
    for sentence in sentences:
        words = simple_tokenize(sentence)
        filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 3]
        tokens.extend(filtered_words)
    
    # Extract numerical conditions
    numerical_conditions = []
    for sentence in sentences:
        # Find patterns like "greater than X", "less than Y", etc.
        patterns = [
            (r'greater than (\d+\.?\d*)', '>', 'float'),
            (r'more than (\d+\.?\d*)', '>', 'float'),
            (r'larger than (\d+\.?\d*)', '>', 'float'),
            (r'less than (\d+\.?\d*)', '<', 'float'),
            (r'smaller than (\d+\.?\d*)', '<', 'float'),
            (r'equal to (\d+\.?\d*)', '=', 'float'),
            (r'equals (\d+\.?\d*)', '=', 'float'),
            (r'between (\d+\.?\d*) and (\d+\.?\d*)', 'BETWEEN', 'range')
        ]
        
        for pattern, operator, value_type in patterns:
            matches = re.findall(pattern, sentence)
            if matches:
                for match in matches:
                    if value_type == 'range' and isinstance(match, tuple) and len(match) == 2:
                        numerical_conditions.append({
                            'operator': operator,
                            'values': [float(match[0]), float(match[1])],
                            'context': sentence
                        })
                    elif value_type == 'float':
                        numerical_conditions.append({
                            'operator': operator,
                            'value': float(match),
                            'context': sentence
                        })
    
    return Dict(dict={
        'tokens': tokens,
        'sentences': sentences,
        'numerical_conditions': numerical_conditions
    })

@calcfunction
def extract_keywords(preprocessed_data):
    """Extract and categorize key terms from preprocessed data."""
    data_dict = preprocessed_data.get_dict()
    tokens = data_dict['tokens']
    
    # Count word frequencies
    word_freq = Counter(tokens)
    top_keywords = {word: count for word, count in word_freq.most_common(10)}
    
    # Identify domain-specific terms
    domain_matches = defaultdict(int)
    for token in tokens:
        for domain, terms in DOMAIN_TERMS.items():
            if token in terms:
                domain_matches[domain] += 1
    
    # Classify the query by domain
    primary_domain = max(domain_matches.items(), key=lambda x: x[1])[0] if domain_matches else 'default'
    
    # Extract potential entity pairs (property-value pairs)
    entity_pairs = []
    for i in range(len(tokens) - 1):
        if tokens[i] in DOMAIN_TERMS['materials'] and tokens[i+1] not in DOMAIN_TERMS['materials']:
            entity_pairs.append((tokens[i], tokens[i+1]))
    
    return Dict(dict={
        'keywords': list(top_keywords.keys()),
        'frequencies': top_keywords,
        'domain': primary_domain,
        'domain_matches': dict(domain_matches),
        'entity_pairs': entity_pairs
    })

@calcfunction
def generate_advanced_query(keywords_dict, preprocessed_data):
    """Generate a domain-specific query from keywords and preprocessed data."""
    keywords = keywords_dict.get_dict()
    preprocessed = preprocessed_data.get_dict()
    
    domain = keywords.get('domain', 'default')
    
    # Prepare the condition parts of the query
    condition_parts = []
    
    # Add keyword conditions (use top 3 keywords)
    top_keywords = keywords['keywords'][:3]
    for kw in top_keywords:
        condition_parts.append(f"CONTAINS('{kw}')")
    
    # Add numerical conditions if found
    numerical_conditions = preprocessed['numerical_conditions']
    for condition in numerical_conditions:
        if condition['operator'] == 'BETWEEN':
            property_candidates = [word for word in condition['context'].split() 
                                  if word in DOMAIN_TERMS.get(domain, [])]
            if property_candidates:
                property_name = property_candidates[0]
                condition_parts.append(
                    f"{property_name} BETWEEN {condition['values'][0]} AND {condition['values'][1]}"
                )
        else:
            property_candidates = [word for word in condition['context'].split() 
                                  if word in DOMAIN_TERMS.get(domain, [])]
            if property_candidates:
                property_name = property_candidates[0]
                condition_parts.append(
                    f"{property_name} {condition['operator']} {condition['value']}"
                )
    
    # Combine all conditions with AND
    conditions = " AND ".join(condition_parts)
    
    # Choose appropriate template
    query_template = QUERY_TEMPLATES.get(domain, QUERY_TEMPLATES['default'])
    query = query_template.format(conditions=conditions)
    
    # Add metadata about the query
    metadata = {
        'domain': domain,
        'complexity': len(condition_parts),
        'keywords_used': top_keywords,
        'numerical_conditions': numerical_conditions
    }
    
    return Dict(dict={'query': query, 'metadata': metadata})

@calcfunction
def validate_query(query_dict):
    """Validate the generated query and suggest improvements."""
    query_data = query_dict.get_dict()
    query = query_data['query']
    metadata = query_data['metadata']
    
    # Simple validation checks
    validation_results = {
        'is_valid': True,
        'suggestions': [],
        'warnings': []
    }
    
    # Check query complexity
    if metadata['complexity'] < 2:
        validation_results['suggestions'].append(
            "Consider adding more specific conditions to narrow down results"
        )
    
    # Check for potential syntax issues
    if 'WHERE' not in query:
        validation_results['is_valid'] = False
        validation_results['warnings'].append("Query is missing the WHERE clause")
    
    # Check for empty conditions
    if 'WHERE ' in query and query.split('WHERE ')[1].strip() == '':
        validation_results['is_valid'] = False
        validation_results['warnings'].append("Query has empty conditions")
    
    return Dict(dict={
        'query': query,
        'metadata': metadata,
        'validation': validation_results
    })

# Helper calcfunctions to create data nodes (important fix)
@calcfunction
def create_string_node(string_value):
    """Create a Str node from a string value."""
    return Str(string_value)

@calcfunction
def create_dict_node(dict_value):
    """Create a Dict node from a dictionary value."""
    return Dict(dict=dict_value)

class SimpleNLQueryWorkChain(WorkChain):
    """Convert natural language to a query without using NLTK."""
    
    @classmethod
    def define(cls, spec):
        super().define(spec)
        # Inputs
        spec.input('description', valid_type=Str, 
                  help='Natural language description')
        spec.input('validate_query', valid_type=Bool, default=lambda: Bool(True),
                  help='Whether to validate the query')
        
        # Outline
        spec.outline(
            cls.preprocess_text,
            cls.extract_keywords,
            cls.generate_query,
            cls.conditionally_validate,
            cls.finalize_results
        )
        
        # Outputs
        spec.output('query', valid_type=Str, 
                   help='Generated query')
        spec.output('analysis', valid_type=Dict, 
                   help='Keyword analysis and domain classification')
        spec.output('metadata', valid_type=Dict, 
                   help='Query metadata and generation info')
        spec.output_namespace('validation', valid_type=Dict, required=False,
                             help='Query validation results')
    
    def preprocess_text(self):
        """First step: Preprocess the input text."""
        self.ctx.preprocessed = preprocess_text(self.inputs.description)
    
    def extract_keywords(self):
        """Second step: Extract and analyze keywords."""
        self.ctx.keywords = extract_keywords(self.ctx.preprocessed)
        self.out('analysis', self.ctx.keywords)
    
    def generate_query(self):
        """Third step: Generate query based on domain and keywords."""
        query_dict = generate_advanced_query(self.ctx.keywords, self.ctx.preprocessed)
        self.ctx.query_data = query_dict
        
        # Extract the query string and use calcfunction to create node
        query_string = query_dict.get_dict()['query']
        self.out('query', create_string_node(query_string))
        
        # Extract metadata and use calcfunction to create node
        metadata_dict = query_dict.get_dict()['metadata']
        self.out('metadata', create_dict_node(metadata_dict))
    
    def conditionally_validate(self):
        """Fourth step: Validate query if requested."""
        if self.inputs.validate_query.value:
            validation_results = validate_query(self.ctx.query_data)
            self.ctx.validation = validation_results
            # Store validation results in output namespace
            self.out('validation', validation_results)
    
    def finalize_results(self):
        """Last step: Log summary of results."""
        self.report(f"Generated query for domain: {self.ctx.keywords.get_dict()['domain']}")
        if self.inputs.validate_query.value:
            validation = self.ctx.validation.get_dict()['validation']
            if not validation['is_valid']:
                self.report(f"Warning: Query validation failed - {validation['warnings']}")
            elif validation['suggestions']:
                self.report(f"Suggestions: {validation['suggestions']}")


if __name__ == "__main__":
    # Import the execute_query function at the top level
    from run_workflow import execute_query
    
    descriptions = [
        "Find all crystal structures with bandgap greater than 2 eV and containing silicon",
        "Show me density functional theory calculations for magnetic materials",
        "I need molecular dynamics simulations of water molecules at high pressure",
        "Get quantum calculations with coherence time greater than 100 microseconds",
        "Find workflows that process crystal structures and run DFT calculations"
    ]

    # Run the workflow for each description
    for desc in descriptions:
        print(f"\n{'='*80}\nProcessing: '{desc}'\n{'='*80}")
        
        # Store description as node before passing to workflow
        desc_node = create_string_node(desc)
        
        result = run(
            SimpleNLQueryWorkChain,
            description=desc_node,
            validate_query=Bool(True)
        )
        
        print("\nDomain Analysis:")
        analysis = result['analysis'].get_dict()
        print(f"  Primary domain: {analysis['domain']}")
        print(f"  Domain matches: {analysis['domain_matches']}")
        print(f"  Top keywords: {', '.join(analysis['keywords'][:5])}")
        
        print("\nGenerated Query:")
        print(f"  {result['query'].value}")
        
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
        
        # Execute the actual database query
        print("\nExecuting Database Query:")
        domain = result['metadata'].get_dict()['domain']
        query_results = execute_query(desc, domain)  # Use the original description for better context
        
        print(f"\nQuery Results ({len(query_results)} matches):")
        
        # Display results in a tabular format
        if query_results:
            # Find all available keys in the results
            all_keys = set()
            for item in query_results:
                all_keys.update(item.keys())
            
            # Filter out common keys we don't need to display
            display_keys = [k for k in all_keys if k not in ['id', 'pk', 'error']]
            
            # Create header with important keys first
            header = ['id', 'pk']
            header.extend(sorted(display_keys))
            
            # Print header
            print(' '.join(f"{key:<15}" for key in header))
            print('-' * (15 * len(header)))
            
            # Print each result
            for item in query_results:
                row = []
                for key in header:
                    value = str(item.get(key, ''))[:14]  # Limit to 14 chars
                    row.append(f"{value:<15}")
                print(' '.join(row))
        else:
            print("  No results found")
        
        # Validate the results against the query criteria
        print("\nResults Validation:")
        
        # Check for common criteria based on the domain
        if domain == 'materials':
            # Count results with silicon (if mentioned in query)
            if 'silicon' in desc.lower():
                silicon_count = sum(1 for r in query_results if 'elements' in r and 'Si' in r.get('elements', []))
                print(f"  Results containing silicon: {silicon_count}/{len(query_results)}")
                
            # Count results with bandgap > 2 (if mentioned in query)
            if 'bandgap' in desc.lower() and 'greater than 2' in desc.lower():
                bandgap_count = sum(1 for r in query_results if 'bandgap' in r and r.get('bandgap', 0) > 2)
                print(f"  Results with bandgap > 2 eV: {bandgap_count}/{len(query_results)}")
                
                # Count results meeting both criteria
                both_criteria = sum(1 for r in query_results 
                                   if 'elements' in r and 'Si' in r.get('elements', []) 
                                   and 'bandgap' in r and r.get('bandgap', 0) > 2)
                print(f"  Results matching BOTH criteria: {both_criteria}/{len(query_results)}")
        
        elif domain == 'quantum':
            # Check coherence time if mentioned
            if 'coherence' in desc.lower() and 'greater than 100' in desc.lower():
                coherence_count = sum(1 for r in query_results if r.get('coherence', 0) > 100)
                print(f"  Results with coherence > 100: {coherence_count}/{len(query_results)}")
        
        print("\nWorkflow Provenance:")
        print(f"  Process PK: {result.pk}")
        print("=" * 80)