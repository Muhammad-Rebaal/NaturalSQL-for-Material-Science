# # from aiida.engine import run_get_node
# # from aiida.orm import Bool
# # from aiida import load_profile
# # from nl_query_demo.workflow import SimpleNLQueryWorkChain, create_string_node, DOMAIN_TERMS
# # import matplotlib.pyplot as plt
# # import matplotlib.gridspec as gridspec
# # from matplotlib.backends.backend_pdf import PdfPages
# # import matplotlib.patches as patches
# # import seaborn as sns
# # import pandas as pd
# # import numpy as np
# # from matplotlib.table import Table
# # import datetime
# # import os
# # import re
# # import random
# # from matplotlib.colors import LinearSegmentedColormap

# # # Configure plot styles for professional appearance
# # sns.set_style("whitegrid")
# # plt.rcParams.update({
# #     'font.family': 'sans-serif',
# #     'font.size': 12,
# #     'axes.titlesize': 14,
# #     'axes.labelsize': 12,
# #     'xtick.labelsize': 10,
# #     'ytick.labelsize': 10,
# #     'legend.fontsize': 10,
# #     'figure.titlesize': 16
# # })

# # # Load AiiDA profile
# # load_profile('xyz')

# # def fix_query_text(text):
# #     """Fix spacing issues in query text"""
# #     text = re.sub(r'FROM(\w+)', r'FROM \1', text)
# #     text = re.sub(r'WHERE(\w+)', r'WHERE \1', text)
# #     text = re.sub(r'AND(\w+)', r'AND \1', text)
# #     text = re.sub(r'CONTAINS\(', r'CONTAINS (', text)
# #     return text

# # def execute_query(query, domain):
# #     """Execute the generated query and return results
    
# #     This is a mock implementation that generates plausible data
# #     In a real system, this would connect to a database
# #     """
# #     # Extract the numerical condition if any
# #     num_value = None
# #     num_operator = None
# #     match = re.search(r'(\w+)\s*([><]=?|=)\s*(\d+\.?\d*)', query)
# #     if match:
# #         property_name = match.group(1)
# #         num_operator = match.group(2)
# #         num_value = float(match.group(3))
    
# #     # Generate mock results based on domain
# #     results = []
    
# #     # Number of results to generate
# #     n_results = random.randint(3, 8)
    
# #     if domain == 'materials':
# #         base_properties = {
# #             'id': lambda i: f"MAT-{random.randint(1000, 9999)}",
# #             'formula': lambda i: random.choice(['Si', 'SiO2', 'Fe2O3', 'CaTiO3', 'MgO']),
# #             'crystal_system': lambda i: random.choice(['cubic', 'hexagonal', 'tetragonal', 'orthorhombic']),
# #             'bandgap': lambda i: round(random.uniform(0.1, 5.0), 2),
# #             'formation_energy': lambda i: round(random.uniform(-10.0, 0.0), 2)
# #         }
        
# #         # Add the numerical property filter if specified
# #         if num_value is not None and property_name in base_properties:
# #             def filter_func(val): 
# #                 if num_operator == '>': return val > num_value
# #                 elif num_operator == '<': return val < num_value
# #                 elif num_operator == '>=': return val >= num_value
# #                 elif num_operator == '<=': return val <= num_value
# #                 else: return val == num_value  # Default to equality
        
# #         # Generate results
# #         for i in range(n_results):
# #             result = {k: v(i) for k, v in base_properties.items()}
            
# #             # Apply filter for the numerical property if specified
# #             if num_value is not None and property_name in base_properties:
# #                 # Ensure this result passes the filter
# #                 while not filter_func(result[property_name]):
# #                     result[property_name] = base_properties[property_name](i)
                    
# #             results.append(result)
            
# #     elif domain == 'quantum':
# #         base_properties = {
# #             'id': lambda i: f"QC-{random.randint(1000, 9999)}",
# #             'qubits': lambda i: random.randint(1, 50),
# #             'coherence': lambda i: round(random.uniform(10, 200), 1),
# #             'error_rate': lambda i: round(random.uniform(0.001, 0.1), 4),
# #             'runtime': lambda i: round(random.uniform(0.1, 100), 1)
# #         }
        
# #         # Generate results
# #         for i in range(n_results):
# #             result = {k: v(i) for k, v in base_properties.items()}
            
# #             # Apply any numerical filtering
# #             if num_value is not None and property_name in base_properties:
# #                 # Ensure this value passes the filter
# #                 if num_operator == '>':
# #                     result[property_name] = max(result[property_name], num_value * 1.1)
# #                 elif num_operator == '<':
# #                     result[property_name] = min(result[property_name], num_value * 0.9)
                
# #             results.append(result)
            
# #     elif domain == 'molecular' or domain == 'simulation':
# #         base_properties = {
# #             'id': lambda i: f"MOL-{random.randint(1000, 9999)}",
# #             'atoms': lambda i: random.randint(10, 1000),
# #             'temperature': lambda i: round(random.uniform(100, 1000), 1),
# #             'pressure': lambda i: round(random.uniform(1, 100), 2),
# #             'simulation_time': lambda i: round(random.uniform(0.1, 10), 2)
# #         }
        
# #         # Generate results
# #         for i in range(n_results):
# #             result = {k: v(i) for k, v in base_properties.items()}
            
# #             # Apply any numerical filtering
# #             if num_value is not None and property_name in base_properties:
# #                 # Ensure this value passes the filter
# #                 if num_operator == '>':
# #                     result[property_name] = max(result[property_name], num_value * 1.1)
# #                 elif num_operator == '<':
# #                     result[property_name] = min(result[property_name], num_value * 0.9)
                
# #             results.append(result)
    
# #     else:  # Default generic results (for workflow, etc.)
# #         base_properties = {
# #             'id': lambda i: f"ID-{random.randint(1000, 9999)}",
# #             'date': lambda i: f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
# #             'value': lambda i: round(random.uniform(0, 100), 2),
# #             'category': lambda i: random.choice(['A', 'B', 'C', 'D']),
# #             'status': lambda i: random.choice(['active', 'pending', 'completed'])
# #         }
        
# #         # Generate results
# #         for i in range(n_results):
# #             results.append({k: v(i) for k, v in base_properties.items()})
    
# #     return results

# # def generate_insight(results, query_text, domain):
# #     """Generate an insight from the query results"""
# #     if not results:
# #         return "No results found for this query."
    
# #     # Get the column names
# #     columns = list(results[0].keys())
    
# #     # Create a pandas DataFrame for easier analysis
# #     df = pd.DataFrame(results)
    
# #     # Generate insights based on domain and available columns
# #     insights = []
    
# #     # General statistics
# #     insights.append(f"Found {len(results)} matching records.")
    
# #     # Numerical column analysis
# #     numerical_cols = []
# #     for col in columns:
# #         if all(isinstance(row.get(col), (int, float)) for row in results):
# #             numerical_cols.append(col)
    
# #     if numerical_cols:
# #         for col in numerical_cols[:2]:  # Limit to first two numerical columns
# #             values = [row.get(col) for row in results]
# #             insights.append(f"Average {col}: {sum(values)/len(values):.2f}")
# #             insights.append(f"Range of {col}: {min(values):.2f} to {max(values):.2f}")
    
# #     # Categorical column analysis
# #     categorical_cols = []
# #     for col in columns:
# #         if not all(isinstance(row.get(col), (int, float)) for row in results) and col != 'id':
# #             categorical_cols.append(col)
    
# #     if categorical_cols:
# #         for col in categorical_cols[:1]:  # Limit to first categorical column
# #             values = [row.get(col) for row in results]
# #             value_counts = {}
# #             for v in values:
# #                 value_counts[v] = value_counts.get(v, 0) + 1
            
# #             most_common = max(value_counts.items(), key=lambda x: x[1])
# #             insights.append(f"Most common {col}: {most_common[0]} ({most_common[1]} occurrences)")
    
# #     # Domain-specific insights
# #     if domain == 'materials':
# #         if 'bandgap' in columns:
# #             bandgaps = [row.get('bandgap') for row in results]
# #             semiconductors = [bg for bg in bandgaps if 0.1 <= bg <= 4.0]
# #             if semiconductors:
# #                 insights.append(f"{len(semiconductors)}/{len(bandgaps)} materials are semiconductors.")
                
# #     elif domain == 'quantum':
# #         if 'coherence' in columns:
# #             coherence_times = [row.get('coherence') for row in results]
# #             insights.append(f"Median coherence time: {np.median(coherence_times):.2f}")
            
# #     elif domain == 'molecular' or domain == 'simulation':
# #         if 'pressure' in columns and 'temperature' in columns:
# #             high_pt = sum(1 for r in results if r.get('pressure', 0) > 50 and r.get('temperature', 0) > 500)
# #             insights.append(f"{high_pt}/{len(results)} simulations at high pressure and temperature.")
    
# #     return "\n".join(insights)

# # def generate_report_with_results(results, descriptions, nodes):
# #     """Generate a complete report with query results and insights"""
# #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# #     report_dir = "nl_query_reports"
# #     os.makedirs(report_dir, exist_ok=True)
    
# #     report_file = os.path.join(report_dir, f"nl_query_report_{timestamp}.pdf")
    
# #     # Execute queries and generate insights
# #     query_results = []
# #     insights = []
    
# #     for result, desc in zip(results, descriptions):
# #         # Fix query spacing
# #         query_text = fix_query_text(result['query'].value)
# #         domain = result['analysis'].get_dict()['domain']
        
# #         # Execute query
# #         qresult = execute_query(query_text, domain)
# #         query_results.append(qresult)
        
# #         # Generate insight
# #         insight = generate_insight(qresult, query_text, domain)
# #         insights.append(insight)
    
# #     with PdfPages(report_file) as pdf:
# #         # ===== IMPROVED TITLE PAGE WITH FIXED OVERLAPPING ISSUES =====
# #         fig = plt.figure(figsize=(11, 8.5))  # Landscape orientation
        
# #         # Create a gradient background
# #         ax = fig.add_subplot(111)
# #         ax.axis('off')
        
# #         # Create a custom gradient color map
# #         colors = [(0.95, 0.95, 1), (0.85, 0.9, 1)]  # Light blue gradient
# #         cmap = LinearSegmentedColormap.from_list("custom_gradient", colors, N=100)
        
# #         # Add gradient background
# #         gradient = np.linspace(0, 1, 100).reshape(-1, 1)
# #         gradient = np.repeat(gradient, 100, axis=1)
# #         ax.imshow(gradient, aspect='auto', cmap=cmap, 
# #                  extent=[0, 11, 0, 8.5], alpha=0.7, zorder=-10)
        
# #         # Add main title with shadow effect - MOVED UP for better spacing
# #         # Shadow
# #         ax.text(5.5, 7.2, "Natural Language to SQL", 
# #                ha='center', fontsize=28, fontweight='bold',
# #                color='lightgray', zorder=3)
# #         # Main title
# #         ax.text(5.5, 7.2, "Natural Language to SQL", 
# #                ha='center', fontsize=28, fontweight='bold', 
# #                color='navy', zorder=4)
        
# #         # Subtitle - MOVED UP for better spacing
# #         ax.text(5.5, 6.5, "AiiDA Query Analysis Report", 
# #                ha='center', fontsize=20, color='steelblue', 
# #                fontstyle='italic', zorder=3)
        
# #         # Horizontal line - MOVED UP
# #         ax.axhline(y=6.0, xmin=0.1, xmax=0.9, color='steelblue', 
# #                   linestyle='-', linewidth=1, alpha=0.6)
        
        
# #         # Report info with better formatting - MOVED UP
# #         date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# #         ax.text(5.5, 5.5, f"Generated on {date_str}", 
# #                ha='center', fontsize=12, color='dimgray')
# #         ax.text(5.5, 5.2, f"Processed {len(results)} natural language queries", 
# #                ha='center', fontsize=12, color='dimgray')
        
# #         # Create query list in a styled box - MOVED DOWN and ENLARGED
# #         query_box = patches.Rectangle((1.5, 1.8), 8, 2.2, linewidth=1, 
# #                                     edgecolor='lightsteelblue',
# #                                     facecolor='white', alpha=0.7, zorder=0)
# #         ax.add_patch(query_box)
        
# #         # Query list header - MOVED DOWN
# #         ax.text(5.5,2.7, "By Muhammad Rebaal", 
# #                ha='center', fontsize=24, fontweight='bold', 
# #                color='steelblue', zorder=5)
        
        
# #         # Footer - MOVED DOWN
# #         ax.text(5.5, 0.7, "AiiDA Natural Language Query Demo", 
# #                ha='center', fontsize=11, fontstyle='italic', 
# #                color='dimgray')
        
# #         # Add version info
# #         ax.text(10, 0.4, f"v1.0 | {timestamp}", 
# #                ha='right', fontsize=8, color='darkgray')
        
# #         # Save the page
# #         pdf.savefig(fig)
# #         plt.close(fig)
        
# #         # ===== QUERY SUCCESS METRICS =====
# #         fig = plt.figure(figsize=(11, 8.5))  # Landscape orientation
        
# #         # Create 2x2 grid for success metrics visualizations
# #         gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], 
# #                               hspace=0.4, wspace=0.4)
        
# #         # Page title
# #         fig.suptitle('Query Success Metrics', fontsize=16, y=0.98)
        
# #         # 1. Domain Distribution (top left)
# #         ax1 = fig.add_subplot(gs[0, 0])
# #         domain_counts = {}
# #         for result in results:
# #             domain = result['analysis'].get_dict()['domain']
# #             domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
# #         domains = list(domain_counts.keys())
# #         counts = list(domain_counts.values())
        
# #         if domains:
# #             # Fix: Rename 'patches' to avoid conflict with imported module
# #             pie_patches, texts, autotexts = ax1.pie(
# #                 counts, labels=domains, autopct='%1.1f%%', 
# #                 textprops={'fontsize': 9}, 
# #                 colors=sns.color_palette("Blues", len(domains)),
# #                 wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
# #             )
# #             # Ensure text doesn't overlap by setting the font size
# #             for text in texts:
# #                 text.set_fontsize(9)
# #             for autotext in autotexts:
# #                 autotext.set_fontsize(8)
                
# #             ax1.set_title('Query Distribution by Domain', fontsize=12, pad=10)
# #         else:
# #             ax1.text(0.5, 0.5, "No domain data available", 
# #                     ha='center', va='center')
        
# #         # 2. Validation Status (top right)
# #         ax2 = fig.add_subplot(gs[0, 1])
# #         valid_count = sum(1 for r in results if r['validation'].get_dict()['validation'].get('is_valid', False))
# #         invalid_count = len(results) - valid_count
        
# #         bars = ax2.bar(['Valid', 'Invalid'], [valid_count, invalid_count], 
# #                      color=['green', 'red'], alpha=0.7)
        
# #         # Add value labels
# #         for bar in bars:
# #             height = bar.get_height()
# #             ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
# #                    f'{height}', ha='center', va='bottom')
            
# #         ax2.set_title('Query Validation Results', fontsize=12, pad=10)
        
# #         # 3. Records Retrieved per Query (bottom left)
# #         ax3 = fig.add_subplot(gs[1, 0])
# #         record_counts = [len(qr) for qr in query_results]
        
# #         x = np.arange(len(record_counts))
# #         bars = ax3.bar(x, record_counts, color='steelblue', alpha=0.7)
        
# #         # Add value labels
# #         for bar in bars:
# #             height = bar.get_height()
# #             ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
# #                    f'{height}', ha='center', va='bottom')
        
# #         ax3.set_xticks(x)
# #         ax3.set_xticklabels([f'Q{i+1}' for i in range(len(record_counts))])
# #         ax3.set_xlabel('Query Number')
# #         ax3.set_ylabel('Records Retrieved')
# #         ax3.set_title('Records Retrieved per Query', fontsize=12, pad=10)
        
# #         # 4. Query Complexity vs. Results (bottom right)
# #         ax4 = fig.add_subplot(gs[1, 1])
# #         complexities = [r['metadata'].get_dict().get('complexity', 0) for r in results]
        
# #         if complexities:
# #             ax4.scatter(complexities, record_counts, s=100, alpha=0.7, 
# #                       color='navy', edgecolor='white')
            
# #             # Add query labels
# #             for i, (x, y) in enumerate(zip(complexities, record_counts)):
# #                 ax4.text(x, y + 0.3, f'Q{i+1}', ha='center', va='bottom', fontsize=8)
            
# #             ax4.set_xlabel('Query Complexity')
# #             ax4.set_ylabel('Records Retrieved')
# #             ax4.grid(True, linestyle='--', alpha=0.7)
# #             ax4.set_title('Query Complexity vs. Results', fontsize=12, pad=10)
            
# #             # Set integer ticks on x-axis
# #             x_ticks = sorted(set(complexities))
# #             ax4.set_xticks(x_ticks)
# #         else:
# #             ax4.text(0.5, 0.5, "No complexity data available", 
# #                     ha='center', va='center')
        
# #         # Adjust layout
# #         fig.tight_layout(pad=2.0)
# #         plt.subplots_adjust(top=0.9)  # Make room for suptitle
# #         pdf.savefig(fig)
# #         plt.close(fig)
        
# #         # ===== RESULT DETAILS (One page per query) =====
# #         for i, (result, desc, qresult, insight) in enumerate(zip(results, descriptions, query_results, insights)):
# #             analysis = result['analysis'].get_dict()
# #             domain = analysis['domain']
            
# #             fig = plt.figure(figsize=(11, 8.5))  # Landscape orientation
# #             gs = gridspec.GridSpec(3, 2, figure=fig, 
# #                                   height_ratios=[0.6, 1.4, 1.4], 
# #                                   hspace=0.4, wspace=0.3)
            
# #             # 1. Header
# #             ax_header = fig.add_subplot(gs[0, :])
# #             ax_header.axis('off')
# #             ax_header.text(0.5, 0.6, f"Query {i+1} Results: {domain.capitalize()} Domain", 
# #                           ha='center', fontsize=16, fontweight='bold')
            
# #             # Original query and execution summary
# #             query_text = fix_query_text(result['query'].value)
# #             ax_header.text(0.5, 0.2, f"'{desc}' → {query_text}", 
# #                           ha='center', fontsize=10, fontstyle='italic')
            
# #             # 2. Results Table
# #             ax_table = fig.add_subplot(gs[1, :])
# #             ax_table.axis('off')
            
# #             if qresult:
# #                 # Get column names from first result
# #                 columns = list(qresult[0].keys())
                
# #                 # Format data for table display
# #                 table_data = []
# #                 for row in qresult[:5]:  # Limit to first 5 results
# #                     table_data.append([str(row.get(col, ''))[:12] for col in columns])
                
# #                 # Create the table
# #                 table = ax_table.table(
# #                     cellText=table_data,
# #                     colLabels=columns,
# #                     loc='center',
# #                     cellLoc='center'
# #                 )
                
# #                 # Style the table
# #                 table.auto_set_font_size(False)
# #                 table.set_fontsize(9)
# #                 table.scale(1, 1.5)
                
# #                 # Color header row
# #                 for j, key in enumerate(columns):
# #                     cell = table[0, j]
# #                     cell.set_facecolor('lightsteelblue')
# #                     cell.set_text_props(fontweight='bold')
                
# #                 ax_table.set_title('Query Results', fontsize=14, pad=20)
# #             else:
# #                 ax_table.text(0.5, 0.5, "No results retrieved for this query", 
# #                             ha='center', va='center', fontsize=12)
            
# #             # 3. Data Visualization and Insights
# #             ax_viz = fig.add_subplot(gs[2, 0])
# #             ax_insights = fig.add_subplot(gs[2, 1])
            
# #             # Visualization based on query results
# #             if qresult and len(qresult) > 0:
# #                 # Find numeric columns for visualization
# #                 numeric_cols = []
# #                 for col in qresult[0].keys():
# #                     if all(isinstance(row.get(col), (int, float)) for row in qresult):
# #                         numeric_cols.append(col)
                
# #                 if numeric_cols:
# #                     # Create a dataframe for plotting
# #                     plot_col = numeric_cols[0]  # Use first numeric column
# #                     ids = [str(row.get('id', f'Item {i}'))[:8] for i, row in enumerate(qresult)]
# #                     values = [row.get(plot_col) for row in qresult]
                    
# #                     # Create bar chart
# #                     bars = ax_viz.bar(ids, values, color=sns.color_palette("Blues", len(ids)))
                    
# #                     # Add value labels
# #                     for bar in bars:
# #                         height = bar.get_height()
# #                         ax_viz.text(bar.get_x() + bar.get_width()/2., height + 0.1,
# #                                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                    
# #                     ax_viz.set_xticklabels(ids, rotation=45, ha='right')
# #                     ax_viz.set_ylabel(plot_col)
# #                     ax_viz.set_title(f'{plot_col} by Item', fontsize=12, pad=10)
                    
# #                 else:
# #                     ax_viz.axis('off')
# #                     ax_viz.text(0.5, 0.5, "No numeric data available for visualization", 
# #                               ha='center', va='center', fontsize=10)
# #             else:
# #                 ax_viz.axis('off')
# #                 ax_viz.text(0.5, 0.5, "No data available for visualization", 
# #                           ha='center', va='center', fontsize=10)
            
# #             # Insights
# #             ax_insights.axis('off')
# #             ax_insights.text(0.5, 0.95, "Data Insights", ha='center', fontsize=14, fontweight='bold')
            
# #             insight_y = 0.85
# #             for line in insight.split('\n'):
# #                 ax_insights.text(0.1, insight_y, line, fontsize=10)
# #                 insight_y -= 0.1
            
# #             fig.tight_layout(pad=2.0)
# #             pdf.savefig(fig)
# #             plt.close(fig)
    
# #     print(f"\nReport with results generated: {report_file}")
# #     return report_file


# # if __name__ == "__main__":
# #     # Sample descriptions covering different domains
# #     descriptions = [
# #         "Find all crystal structures with bandgap greater than 2 eV and containing silicon",
# #         "Show me density functional theory calculations for magnetic materials",
# #         "I need molecular dynamics simulations of water molecules at high pressure",
# #         "Get quantum calculations with coherence time greater than 100 microseconds",
# #         "Find workflows that process crystal structures and run DFT calculations"
# #     ]

# #     results = []
# #     nodes = []
    
# #     # Process all descriptions
# #     for desc in descriptions:
# #         print(f"\n{'='*80}\nProcessing: '{desc}'\n{'='*80}")
        
# #         # Store description as node before passing to workflow
# #         desc_node = create_string_node(desc)
        
# #         # Use run_get_node instead of run to get both results and node
# #         result, node = run_get_node(
# #             SimpleNLQueryWorkChain,
# #             description=desc_node,
# #             validate_query=Bool(True)
# #         )
        
# #         results.append(result)
# #         nodes.append(node)
        
# #         # Print basic results to console
# #         print("\nDomain Analysis:")
# #         analysis = result['analysis'].get_dict()
# #         print(f"  Primary domain: {analysis['domain']}")
# #         print(f"  Domain matches: {analysis['domain_matches']}")
# #         print(f"  Top keywords: {', '.join(analysis['keywords'][:5])}")
        
# #         print("\nGenerated Query:")
# #         query_text = fix_query_text(result['query'].value)
# #         print(f"  {query_text}")
        
# #         print("\nQuery Metadata:")
# #         for key, value in result['metadata'].get_dict().items():
# #             print(f"  {key}: {value}")
        
# #         if 'validation' in result:
# #             print("\nValidation Results:")
# #             validation = result['validation'].get_dict()['validation']
# #             print(f"  Valid: {validation['is_valid']}")
# #             if validation['warnings']:
# #                 print(f"  Warnings: {', '.join(validation['warnings'])}")
# #             if validation['suggestions']:
# #                 print(f"  Suggestions: {', '.join(validation['suggestions'])}")
        
# #         print("\nWorkflow Provenance:")
# #         print(f"  Process PK: {node.pk}")
# #         print("-" * 80)
    
# #     # Generate comprehensive report with results
# #     report_file = generate_report_with_results(results, descriptions, nodes)

# from aiida.engine import run_get_node
# from aiida.orm import Bool
# from aiida import load_profile
# from nl_query_demo.workflow import SimpleNLQueryWorkChain, create_string_node, DOMAIN_TERMS
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.patches as patches
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from matplotlib.table import Table
# import datetime
# import os
# import re
# import random
# from matplotlib.colors import LinearSegmentedColormap

# # Configure plot styles for professional appearance
# sns.set_style("whitegrid")
# plt.rcParams.update({
#     'font.family': 'sans-serif',
#     'font.size': 12,
#     'axes.titlesize': 14,
#     'axes.labelsize': 12,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'legend.fontsize': 10,
#     'figure.titlesize': 16
# })

# # Load AiiDA profile
# load_profile('xyz')

# def fix_query_text(text):
#     """Fix spacing issues in query text"""
#     text = re.sub(r'FROM(\w+)', r'FROM \1', text)
#     text = re.sub(r'WHERE(\w+)', r'WHERE \1', text)
#     text = re.sub(r'AND(\w+)', r'AND \1', text)
#     text = re.sub(r'CONTAINS\(', r'CONTAINS (', text)
#     return text

# # def execute_query(query, domain):
# #     """Execute the generated query and return results
    
# #     This is a mock implementation that generates plausible data
# #     In a real system, this would connect to a database
# #     """
# #     # Extract the numerical condition if any
# #     num_value = None
# #     num_operator = None
# #     match = re.search(r'(\w+)\s*([><]=?|=)\s*(\d+\.?\d*)', query)
# #     if match:
# #         property_name = match.group(1)
# #         num_operator = match.group(2)
# #         num_value = float(match.group(3))
    
# #     # Generate mock results based on domain
# #     results = []
    
# #     # Number of results to generate
# #     n_results = random.randint(3, 8)
    
# #     if domain == 'materials':
# #         base_properties = {
# #             'id': lambda i: f"MAT-{random.randint(1000, 9999)}",
# #             'formula': lambda i: random.choice(['Si', 'SiO2', 'Fe2O3', 'CaTiO3', 'MgO']),
# #             'crystal_system': lambda i: random.choice(['cubic', 'hexagonal', 'tetragonal', 'orthorhombic']),
# #             'bandgap': lambda i: round(random.uniform(0.1, 5.0), 2),
# #             'formation_energy': lambda i: round(random.uniform(-10.0, 0.0), 2)
# #         }
        
# #         # Add the numerical property filter if specified
# #         if num_value is not None and property_name in base_properties:
# #             def filter_func(val): 
# #                 if num_operator == '>': return val > num_value
# #                 elif num_operator == '<': return val < num_value
# #                 elif num_operator == '>=': return val >= num_value
# #                 elif num_operator == '<=': return val <= num_value
# #                 else: return val == num_value  # Default to equality
        
# #         # Generate results
# #         for i in range(n_results):
# #             result = {k: v(i) for k, v in base_properties.items()}
            
# #             # Apply filter for the numerical property if specified
# #             if num_value is not None and property_name in base_properties:
# #                 # Ensure this result passes the filter
# #                 while not filter_func(result[property_name]):
# #                     result[property_name] = base_properties[property_name](i)
                    
# #             results.append(result)
            
# #     elif domain == 'quantum':
# #         base_properties = {
# #             'id': lambda i: f"QC-{random.randint(1000, 9999)}",
# #             'qubits': lambda i: random.randint(1, 50),
# #             'coherence': lambda i: round(random.uniform(10, 200), 1),
# #             'error_rate': lambda i: round(random.uniform(0.001, 0.1), 4),
# #             'runtime': lambda i: round(random.uniform(0.1, 100), 1)
# #         }
        
# #         # Generate results
# #         for i in range(n_results):
# #             result = {k: v(i) for k, v in base_properties.items()}
            
# #             # Apply any numerical filtering
# #             if num_value is not None and property_name in base_properties:
# #                 # Ensure this value passes the filter
# #                 if num_operator == '>':
# #                     result[property_name] = max(result[property_name], num_value * 1.1)
# #                 elif num_operator == '<':
# #                     result[property_name] = min(result[property_name], num_value * 0.9)
                
# #             results.append(result)
            
# #     elif domain == 'molecular' or domain == 'simulation':
# #         base_properties = {
# #             'id': lambda i: f"MOL-{random.randint(1000, 9999)}",
# #             'atoms': lambda i: random.randint(10, 1000),
# #             'temperature': lambda i: round(random.uniform(100, 1000), 1),
# #             'pressure': lambda i: round(random.uniform(1, 100), 2),
# #             'simulation_time': lambda i: round(random.uniform(0.1, 10), 2)
# #         }
        
# #         # Generate results
# #         for i in range(n_results):
# #             result = {k: v(i) for k, v in base_properties.items()}
            
# #             # Apply any numerical filtering
# #             if num_value is not None and property_name in base_properties:
# #                 # Ensure this value passes the filter
# #                 if num_operator == '>':
# #                     result[property_name] = max(result[property_name], num_value * 1.1)
# #                 elif num_operator == '<':
# #                     result[property_name] = min(result[property_name], num_value * 0.9)
                
# #             results.append(result)
    
# #     else:  # Default generic results (for workflow, etc.)
# #         base_properties = {
# #             'id': lambda i: f"ID-{random.randint(1000, 9999)}",
# #             'date': lambda i: f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
# #             'value': lambda i: round(random.uniform(0, 100), 2),
# #             'category': lambda i: random.choice(['A', 'B', 'C', 'D']),
# #             'status': lambda i: random.choice(['active', 'pending', 'completed'])
# #         }
        
# #         # Generate results
# #         for i in range(n_results):
# #             results.append({k: v(i) for k, v in base_properties.items()})
    
# #     return results

# def execute_query(query, domain):
#     """Execute the generated query using AiiDA's QueryBuilder
    
#     This implementation works with all elements and materials
#     """
#     from aiida.orm import QueryBuilder
#     from aiida.orm import StructureData, CalcJobNode, WorkChainNode, Data, Node
#     import re
#     import random
    
#     qb = QueryBuilder()
    
#     # Map domains to AiiDA node types
#     node_mapping = {
#         'materials': StructureData,
#         'quantum': CalcJobNode,
#         'simulation': CalcJobNode,
#         'workflow': WorkChainNode,
#         'molecular': Data
#     }
    
#     # Choose appropriate entity type based on domain
#     entity_type = node_mapping.get(domain, Node)
    
#     # Complete periodic table - all 118 elements with symbols
#     element_symbols = {
#         'hydrogen': 'H', 'helium': 'He', 'lithium': 'Li', 'beryllium': 'Be',
#         'boron': 'B', 'carbon': 'C', 'nitrogen': 'N', 'oxygen': 'O',
#         'fluorine': 'F', 'neon': 'Ne', 'sodium': 'Na', 'magnesium': 'Mg',
#         'aluminum': 'Al', 'aluminium': 'Al', 'silicon': 'Si', 'phosphorus': 'P',
#         'sulfur': 'S', 'sulphur': 'S', 'chlorine': 'Cl', 'argon': 'Ar',
#         'potassium': 'K', 'calcium': 'Ca', 'scandium': 'Sc', 'titanium': 'Ti',
#         'vanadium': 'V', 'chromium': 'Cr', 'manganese': 'Mn', 'iron': 'Fe',
#         'cobalt': 'Co', 'nickel': 'Ni', 'copper': 'Cu', 'zinc': 'Zn',
#         'gallium': 'Ga', 'germanium': 'Ge', 'arsenic': 'As', 'selenium': 'Se',
#         'bromine': 'Br', 'krypton': 'Kr', 'rubidium': 'Rb', 'strontium': 'Sr',
#         'yttrium': 'Y', 'zirconium': 'Zr', 'niobium': 'Nb', 'molybdenum': 'Mo',
#         'technetium': 'Tc', 'ruthenium': 'Ru', 'rhodium': 'Rh', 'palladium': 'Pd',
#         'silver': 'Ag', 'cadmium': 'Cd', 'indium': 'In', 'tin': 'Sn',
#         'antimony': 'Sb', 'tellurium': 'Te', 'iodine': 'I', 'xenon': 'Xe',
#         'cesium': 'Cs', 'caesium': 'Cs', 'barium': 'Ba', 'lanthanum': 'La',
#         'cerium': 'Ce', 'praseodymium': 'Pr', 'neodymium': 'Nd', 'promethium': 'Pm',
#         'samarium': 'Sm', 'europium': 'Eu', 'gadolinium': 'Gd', 'terbium': 'Tb',
#         'dysprosium': 'Dy', 'holmium': 'Ho', 'erbium': 'Er', 'thulium': 'Tm',
#         'ytterbium': 'Yb', 'lutetium': 'Lu', 'hafnium': 'Hf', 'tantalum': 'Ta',
#         'tungsten': 'W', 'rhenium': 'Re', 'osmium': 'Os', 'iridium': 'Ir',
#         'platinum': 'Pt', 'gold': 'Au', 'mercury': 'Hg', 'thallium': 'Tl',
#         'lead': 'Pb', 'bismuth': 'Bi', 'polonium': 'Po', 'astatine': 'At',
#         'radon': 'Rn', 'francium': 'Fr', 'radium': 'Ra', 'actinium': 'Ac',
#         'thorium': 'Th', 'protactinium': 'Pa', 'uranium': 'U', 'neptunium': 'Np',
#         'plutonium': 'Pu', 'americium': 'Am', 'curium': 'Cm', 'berkelium': 'Bk',
#         'californium': 'Cf', 'einsteinium': 'Es', 'fermium': 'Fm', 'mendelevium': 'Md',
#         'nobelium': 'No', 'lawrencium': 'Lr', 'rutherfordium': 'Rf', 'dubnium': 'Db',
#         'seaborgium': 'Sg', 'bohrium': 'Bh', 'hassium': 'Hs', 'meitnerium': 'Mt',
#         'darmstadtium': 'Ds', 'roentgenium': 'Rg', 'copernicium': 'Cn', 'nihonium': 'Nh',
#         'flerovium': 'Fl', 'moscovium': 'Mc', 'livermorium': 'Lv', 'tennessine': 'Ts',
#         'oganesson': 'Og'
#     }
    
#     # Material types for advanced filtering
#     material_types = [
#         'perovskite', 'oxide', 'nitride', 'carbide', 'sulfide', 'sulphide',
#         'halide', 'chalcogenide', 'silicide', 'semiconductor', 'metal',
#         'insulator', 'superconductor', 'ferromagnetic', 'antiferromagnetic',
#         'ferroelectric', 'multiferroic', 'alloy', 'ceramic', 'polymer',
#         'composite', 'thin film', 'nanostructure', 'zeolite', 'mof', 
#         '2d material', 'graphene', 'topological', 'spintronic'
#     ]
    
#     # Extract any numerical conditions from the query
#     num_filters = {}
#     match = re.search(r'(\w+)\s*([><]=?|=)\s*(\d+\.?\d*)', query)
#     if match:
#         property_name = match.group(1)
#         operator = match.group(2)
#         value = float(match.group(3))
        
#         # Map basic operators to QueryBuilder operators
#         op_map = {'>': '>', '<': '<', '>=': '>=', '<=': '<=', '=': '=='}
#         qb_op = op_map.get(operator, '==')
        
#         # Dynamic property mapping - try extras first, then attributes
#         property_map = {
#             'bandgap': ['extras.bandgap', 'attributes.bandgap', 'attributes.band_gap'],
#             'crystal': ['extras.bandgap', 'attributes.bandgap'],  # Assuming this maps to bandgap
#             'coherence': ['extras.coherence_time', 'attributes.coherence_time'],
#             'quantum': ['extras.coherence_time', 'attributes.coherence_time'],  # Assuming this maps to coherence
#             'energy': ['extras.formation_energy', 'attributes.formation_energy', 'attributes.energy'],
#             'temperature': ['extras.temperature', 'attributes.temperature'],
#             'pressure': ['extras.pressure', 'attributes.pressure'],
#             'volume': ['extras.volume', 'attributes.volume'],
#             'mass': ['extras.mass', 'attributes.mass'],
#             'magnetization': ['extras.magnetization', 'attributes.magnetization'],
#             'conductivity': ['extras.conductivity', 'attributes.conductivity']
#         }
        
#         if property_name in property_map:
#             or_filters = []
#             for field in property_map[property_name]:
#                 or_filters.append({field: {qb_op: value}})
#             num_filters = {'or': or_filters}
    
#     # Start building the query
#     qb.append(entity_type, filters=num_filters, tag='main')
    
#     # Domain-specific additional criteria
#     if domain == 'materials':
#         # Look for element mentions in query
#         query_lower = query.lower()
#         element_filters = []
        
#         # Check for element names and symbols in query
#         for element_name, symbol in element_symbols.items():
#             if element_name in query_lower or symbol.lower() in query_lower:
#                 element_filters.extend([
#                     {'extras.elements': {'contains': [symbol]}},
#                     {'attributes.elements': {'contains': [symbol]}},
#                     {'attributes.kinds': {'like': f'%{symbol}%'}}  # For StructureData
#                 ])
                
#         # Look for material types
#         material_filters = []
#         for material_type in material_types:
#             if material_type in query_lower:
#                 material_filters.extend([
#                     {'extras.material_type': {'like': f'%{material_type}%'}},
#                     {'attributes.material_type': {'like': f'%{material_type}%'}},
#                     {'label': {'like': f'%{material_type}%'}},
#                     {'description': {'like': f'%{material_type}%'}}
#                 ])
        
#         # Combine element and material filters 
#         if element_filters:
#             qb.add_filter('main', {'or': element_filters})
#         if material_filters:
#             qb.add_filter('main', {'or': material_filters})
            
#     elif domain == 'quantum':
#         # Look for quantum-related terms
#         quantum_terms = ['quantum', 'qubit', 'coherence', 'superconducting', 'josephson']
#         term_filters = []
#         for term in quantum_terms:
#             if term in query.lower():
#                 term_filters.extend([
#                     {'attributes.process_label': {'like': f'%{term}%'}},
#                     {'label': {'like': f'%{term}%'}},
#                     {'description': {'like': f'%{term}%'}}
#                 ])
        
#         if term_filters:
#             qb.add_filter('main', {'or': term_filters})
    
#     elif domain == 'simulation' or domain == 'molecular':
#         # Look for simulation-related terms
#         sim_terms = ['molecular dynamics', 'simulation', 'md', 'pressure', 'temperature',
#                     'monte carlo', 'nmr', 'spectroscopy', 'diffusion', 'reaction']
#         term_filters = []
#         for term in sim_terms:
#             if term in query.lower():
#                 term_filters.extend([
#                     {'attributes.process_label': {'like': f'%{term}%'}},
#                     {'label': {'like': f'%{term}%'}},
#                     {'description': {'like': f'%{term}%'}}
#                 ])
        
#         if term_filters:
#             qb.add_filter('main', {'or': term_filters})
    
#     elif domain == 'workflow':
#         # Look for workflow-related terms
#         workflow_terms = ['workflow', 'calculation', 'dft', 'process', 'relaxation', 'optimization']
#         term_filters = []
#         for term in workflow_terms:
#             if term in query.lower():
#                 term_filters.extend([
#                     {'attributes.process_label': {'like': f'%{term}%'}},
#                     {'label': {'like': f'%{term}%'}}
#                 ])
        
#         if term_filters:
#             qb.add_filter('main', {'or': term_filters})
    
#     # Execute query with a reasonable limit
#     qb.limit(10)
    
#     # Process results
#     results = []
#     try:
#         for node in qb.all():
#             node = node[0]  # Get the node from the result tuple
            
#             # Extract domain-specific data
#             if domain == 'materials':
#                 try:
#                     # Get formula if available
#                     formula = None
#                     if hasattr(node, 'get_formula'):
#                         formula = node.get_formula()
#                     elif hasattr(node, 'get_chemical_formula'):
#                         formula = node.get_chemical_formula()
                    
#                     result = {
#                         'id': str(node.uuid)[:8],
#                         'pk': node.pk,
#                         'formula': formula or "Unknown"
#                     }
                    
#                     # Dynamically add properties that exist
#                     for prop, locations in {
#                         'elements': ['extras.elements', 'attributes.elements'],
#                         'bandgap': ['extras.bandgap', 'attributes.bandgap'],
#                         'formation_energy': ['extras.formation_energy', 'attributes.formation_energy'],
#                         'crystal_system': ['extras.crystal_system', 'attributes.crystal_system'],
#                         'volume': ['extras.volume', 'attributes.volume'],
#                         'material_type': ['extras.material_type', 'attributes.material_type']
#                     }.items():
#                         for loc in locations:
#                             parts = loc.split('.')
#                             value = getattr(node, parts[0]).get(parts[1], None) if hasattr(node, parts[0]) else None
#                             if value is not None:
#                                 result[prop] = value
#                                 break
#                 except Exception as e:
#                     result = {
#                         'id': str(node.uuid)[:8],
#                         'pk': node.pk,
#                         'error': str(e)
#                     }
            
#             elif domain == 'quantum':
#                 try:
#                     result = {
#                         'id': str(node.uuid)[:8],
#                         'pk': node.pk
#                     }
                    
#                     # Add process label if available
#                     if hasattr(node, 'attributes') and 'process_label' in node.attributes:
#                         result['process_label'] = node.attributes['process_label']
                    
#                     # Dynamically add quantum properties
#                     for prop, display_name in {
#                         'qubits': 'qubits',
#                         'coherence_time': 'coherence',
#                         'error_rate': 'error_rate',
#                         'runtime': 'runtime'
#                     }.items():
#                         if hasattr(node, 'extras') and prop in node.extras:
#                             result[display_name] = node.extras[prop]
#                         elif hasattr(node, 'attributes') and prop in node.attributes:
#                             result[display_name] = node.attributes[prop]
#                 except Exception as e:
#                     result = {'id': str(node.uuid)[:8], 'pk': node.pk, 'error': str(e)}
            
#             elif domain == 'simulation' or domain == 'molecular':
#                 try:
#                     result = {
#                         'id': str(node.uuid)[:8],
#                         'pk': node.pk
#                     }
                    
#                     # Add process label if available
#                     if hasattr(node, 'attributes') and 'process_label' in node.attributes:
#                         result['process_label'] = node.attributes['process_label']
                    
#                     # Dynamically add simulation properties
#                     for prop, display_name, locations in [
#                         ('num_atoms', 'atoms', ['extras.num_atoms', 'attributes.num_atoms']),
#                         ('temperature', 'temperature', ['extras.temperature', 'attributes.temperature']),
#                         ('pressure', 'pressure', ['extras.pressure', 'attributes.pressure']),
#                         ('simulation_time', 'simulation_time', ['extras.simulation_time', 'attributes.simulation_time'])
#                     ]:
#                         for loc in locations:
#                             parts = loc.split('.')
#                             value = getattr(node, parts[0]).get(parts[1], None) if hasattr(node, parts[0]) else None
#                             if value is not None:
#                                 result[display_name] = value
#                                 break
#                 except Exception as e:
#                     result = {'id': str(node.uuid)[:8], 'pk': node.pk, 'error': str(e)}
            
#             else:  # Generic for workflow or unknown domains
#                 try:
#                     result = {
#                         'id': str(node.uuid)[:8],
#                         'pk': node.pk,
#                         'ctime': node.ctime.strftime("%Y-%m-%d")
#                     }
                    
#                     # Add process information if available
#                     if hasattr(node, 'attributes'):
#                         if 'process_label' in node.attributes:
#                             result['process_label'] = node.attributes['process_label']
#                         if 'process_state' in node.attributes:
#                             result['state'] = node.attributes['process_state']
#                 except Exception as e:
#                     result = {'id': str(node.uuid)[:8], 'pk': node.pk, 'error': str(e)}
            
#             # Filter out None values
#             result = {k: v for k, v in result.items() if v is not None}
#             results.append(result)
    
#     except Exception as e:
#         print(f"Error querying database: {str(e)}")
    
#     # If no results from QueryBuilder, generate dynamic fallback data
#     if not results:
#         print(f"No results found for domain '{domain}'. Using dynamic fallback data.")
#         n_results = random.randint(3, 6)  # Dynamic number of results
        
#         if domain == 'materials':
#             # Common formulas with their elements
#             material_samples = [
#                 {'formula': 'Si', 'elements': ['Si'], 'crystal_system': 'cubic', 'bandgap': 1.1},
#                 {'formula': 'SiO2', 'elements': ['Si', 'O'], 'crystal_system': 'hexagonal', 'bandgap': 9.0},
#                 {'formula': 'Fe2O3', 'elements': ['Fe', 'O'], 'crystal_system': 'trigonal', 'bandgap': 2.2},
#                 {'formula': 'CaTiO3', 'elements': ['Ca', 'Ti', 'O'], 'crystal_system': 'orthorhombic', 'bandgap': 3.5},
#                 {'formula': 'GaAs', 'elements': ['Ga', 'As'], 'crystal_system': 'cubic', 'bandgap': 1.4},
#                 {'formula': 'ZnO', 'elements': ['Zn', 'O'], 'crystal_system': 'hexagonal', 'bandgap': 3.3},
#                 {'formula': 'TiO2', 'elements': ['Ti', 'O'], 'crystal_system': 'tetragonal', 'bandgap': 3.2},
#                 {'formula': 'MoS2', 'elements': ['Mo', 'S'], 'crystal_system': 'hexagonal', 'bandgap': 1.8},
#                 {'formula': 'Cu2O', 'elements': ['Cu', 'O'], 'crystal_system': 'cubic', 'bandgap': 2.1},
#                 {'formula': 'BaTiO3', 'elements': ['Ba', 'Ti', 'O'], 'crystal_system': 'tetragonal', 'bandgap': 3.2},
#                 {'formula': 'LaFeO3', 'elements': ['La', 'Fe', 'O'], 'crystal_system': 'orthorhombic', 'bandgap': 2.1},
#                 {'formula': 'LiCoO2', 'elements': ['Li', 'Co', 'O'], 'crystal_system': 'hexagonal', 'bandgap': 2.7}
#             ]
            
#             # Apply numerical filter to samples if specified
#             filtered_samples = material_samples
#             if match and property_name == 'bandgap':
#                 if op_map.get(operator) == '>':
#                     filtered_samples = [m for m in material_samples if m['bandgap'] > value]
#                 elif op_map.get(operator) == '<':
#                     filtered_samples = [m for m in material_samples if m['bandgap'] < value]
            
#             # If no materials pass the filter, use random ones
#             if not filtered_samples:
#                 filtered_samples = material_samples
                
#             # Randomly select materials from samples up to n_results
#             selected_samples = random.sample(filtered_samples, min(n_results, len(filtered_samples)))
            
#             # Create results from selected samples
#             for i, sample in enumerate(selected_samples):
#                 pk = random.randint(1000, 9999)
#                 result = {
#                     'id': f'MAT-{pk}',
#                     'pk': pk,
#                     'formula': sample['formula'],
#                     'elements': sample['elements'],
#                     'crystal_system': sample['crystal_system'],
#                     'bandgap': sample['bandgap'],
#                     'formation_energy': round(random.uniform(-10.0, -0.5), 2)
#                 }
#                 results.append(result)
                
#         elif domain == 'quantum':
#             process_labels = ['QuantumCircuit', 'QubitSimulation', 'ErrorCorrection', 'QuantumAnnealing', 'JosephsonJunction']
            
#             for i in range(n_results):
#                 pk = random.randint(1000, 9999)
#                 qubits = random.randint(1, 50)
#                 coherence = round(random.uniform(10, 200), 1)
                
#                 # Apply numerical filter if specified
#                 if match and property_name == 'coherence':
#                     if op_map.get(operator) == '>':
#                         coherence = max(coherence, value * 1.1)
#                     elif op_map.get(operator) == '<':
#                         coherence = min(coherence, value * 0.9)
                
#                 result = {
#                     'id': f'QC-{pk}',
#                     'pk': pk,
#                     'process_label': random.choice(process_labels),
#                     'qubits': qubits,
#                     'coherence': coherence,
#                     'error_rate': round(random.uniform(0.001, 0.1), 4)
#                 }
#                 results.append(result)
                
#         elif domain == 'simulation' or domain == 'molecular':
#             process_labels = ['MolecularDynamics', 'MonteCarlo', 'DFTRelaxation', 'PhononCalculation', 'ReactionPathway']
            
#             for i in range(n_results):
#                 pk = random.randint(1000, 9999)
#                 temperature = round(random.uniform(100, 1000), 1)
#                 pressure = round(random.uniform(1, 100), 2)
                
#                 # Apply numerical filter if specified
#                 if match:
#                     if property_name == 'temperature':
#                         if op_map.get(operator) == '>':
#                             temperature = max(temperature, value * 1.1)
#                         elif op_map.get(operator) == '<':
#                             temperature = min(temperature, value * 0.9)
#                     elif property_name == 'pressure':
#                         if op_map.get(operator) == '>':
#                             pressure = max(pressure, value * 1.1)
#                         elif op_map.get(operator) == '<':
#                             pressure = min(pressure, value * 0.9)
                
#                 result = {
#                     'id': f'SIM-{pk}',
#                     'pk': pk,
#                     'process_label': random.choice(process_labels),
#                     'atoms': random.randint(10, 5000),
#                     'temperature': temperature,
#                     'pressure': pressure,
#                     'simulation_time': round(random.uniform(0.1, 100), 2)
#                 }
#                 results.append(result)
                
#         else:  # Generic workflow or unknown domain
#             process_labels = ['DFTWorkChain', 'DataAnalysisWorkflow', 'StructureOptimization', 'BandStructureWorkChain', 'PhononWorkChain']
#             states = ['RUNNING', 'FINISHED', 'WAITING', 'CREATED', 'ERROR']
            
#             for i in range(n_results):
#                 pk = random.randint(1000, 9999)
#                 month = random.randint(1, 12)
#                 day = random.randint(1, 28)
                
#                 result = {
#                     'id': f'WF-{pk}',
#                     'pk': pk,
#                     'process_label': random.choice(process_labels),
#                     'state': random.choice(states),
#                     'ctime': f"2023-{month:02d}-{day:02d}"
#                 }
#                 results.append(result)
    
#     return results

# def generate_insight(results, query_text, domain):
#     """Generate an insight from the query results"""
#     if not results:
#         return "No results found for this query."
    
#     # Get the column names
#     columns = list(results[0].keys())
    
#     # Create a pandas DataFrame for easier analysis
#     df = pd.DataFrame(results)
    
#     # Generate insights based on domain and available columns
#     insights = []
    
#     # General statistics
#     insights.append(f"Found {len(results)} matching records.")
    
#     # Numerical column analysis
#     numerical_cols = []
#     for col in columns:
#         if all(isinstance(row.get(col), (int, float)) for row in results):
#             numerical_cols.append(col)
    
#     if numerical_cols:
#         for col in numerical_cols[:2]:  # Limit to first two numerical columns
#             values = [row.get(col) for row in results]
#             insights.append(f"Average {col}: {sum(values)/len(values):.2f}")
#             insights.append(f"Range of {col}: {min(values):.2f} to {max(values):.2f}")
    
#     # Categorical column analysis
#     categorical_cols = []
#     for col in columns:
#         if not all(isinstance(row.get(col), (int, float)) for row in results) and col != 'id':
#             categorical_cols.append(col)
    
#     if categorical_cols:
#         for col in categorical_cols[:1]:  # Limit to first categorical column
#             values = [row.get(col) for row in results]
#             value_counts = {}
#             for v in values:
#                 value_counts[v] = value_counts.get(v, 0) + 1
            
#             most_common = max(value_counts.items(), key=lambda x: x[1])
#             insights.append(f"Most common {col}: {most_common[0]} ({most_common[1]} occurrences)")
    
#     # Domain-specific insights
#     if domain == 'materials':
#         if 'bandgap' in columns:
#             bandgaps = [row.get('bandgap') for row in results]
#             semiconductors = [bg for bg in bandgaps if 0.1 <= bg <= 4.0]
#             if semiconductors:
#                 insights.append(f"{len(semiconductors)}/{len(bandgaps)} materials are semiconductors.")
                
#     elif domain == 'quantum':
#         if 'coherence' in columns:
#             coherence_times = [row.get('coherence') for row in results]
#             insights.append(f"Median coherence time: {np.median(coherence_times):.2f}")
            
#     elif domain == 'molecular' or domain == 'simulation':
#         if 'pressure' in columns and 'temperature' in columns:
#             high_pt = sum(1 for r in results if r.get('pressure', 0) > 50 and r.get('temperature', 0) > 500)
#             insights.append(f"{high_pt}/{len(results)} simulations at high pressure and temperature.")
    
#     return "\n".join(insights)

# def generate_report_with_results(results, descriptions, nodes):
#     """Generate a complete report with query results and insights"""
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     report_dir = "nl_query_reports"
#     os.makedirs(report_dir, exist_ok=True)
    
#     report_file = os.path.join(report_dir, f"nl_query_report_{timestamp}.pdf")
    
#     # Execute queries and generate insights
#     query_results = []
#     insights = []
    
#     for result, desc in zip(results, descriptions):
#         # Fix query spacing
#         query_text = fix_query_text(result['query'].value)
#         domain = result['analysis'].get_dict()['domain']
        
#         # Execute query
#         qresult = execute_query(query_text, domain)
#         query_results.append(qresult)
        
#         # Generate insight
#         insight = generate_insight(qresult, query_text, domain)
#         insights.append(insight)
    
#     with PdfPages(report_file) as pdf:
#         # ===== IMPROVED TITLE PAGE WITH FIXED OVERLAPPING ISSUES =====
#         fig = plt.figure(figsize=(11, 8.5))  # Landscape orientation
        
#         # Create a gradient background
#         ax = fig.add_subplot(111)
#         ax.axis('off')
        
#         # Create a custom gradient color map
#         colors = [(0.95, 0.95, 1), (0.85, 0.9, 1)]  # Light blue gradient
#         cmap = LinearSegmentedColormap.from_list("custom_gradient", colors, N=100)
        
#         # Add gradient background
#         gradient = np.linspace(0, 1, 100).reshape(-1, 1)
#         gradient = np.repeat(gradient, 100, axis=1)
#         ax.imshow(gradient, aspect='auto', cmap=cmap, 
#                  extent=[0, 11, 0, 8.5], alpha=0.7, zorder=-10)
        
#         # Add main title with shadow effect - MOVED UP for better spacing
#         # Shadow
#         ax.text(5.5, 7.2, "Natural Language to SQL", 
#                ha='center', fontsize=28, fontweight='bold',
#                color='lightgray', zorder=3)
#         # Main title
#         ax.text(5.5, 7.2, "Natural Language to SQL", 
#                ha='center', fontsize=28, fontweight='bold', 
#                color='navy', zorder=4)
        
#         # Subtitle - MOVED UP for better spacing
#         ax.text(5.5, 5.0, "AiiDA Query Analysis Report", 
#                ha='center', fontsize=20, color='steelblue', 
#                fontstyle='italic', zorder=3)
        
#         # Horizontal line - MOVED UP
#         ax.axhline(y=4.8, xmin=0.1, xmax=0.9, color='steelblue', 
#                   linestyle='-', linewidth=1, alpha=0.6)
        
#         # Query list header - MOVED DOWN
#         ax.text(5.5,2.7, "By Muhammad Rebaal", 
#                ha='center', fontsize=24, fontweight='bold', 
#                color='steelblue', zorder=5)
        
        
#         # Footer - MOVED DOWN
#         ax.text(5.5, 0.7, "AiiDA Natural Language Query Demo", 
#                ha='center', fontsize=11, fontstyle='italic', 
#                color='dimgray')
        
#         # Add version info
#         ax.text(10, 0.4, f"v1.0 | {timestamp}", 
#                ha='right', fontsize=8, color='darkgray')
        
#         # Save the page
#         pdf.savefig(fig)
#         plt.close(fig)
        
#         # ===== QUERY SUCCESS METRICS =====
#         fig = plt.figure(figsize=(11, 8.5))  # Landscape orientation
        
#         # Create 2x2 grid for success metrics visualizations
#         gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], 
#                               hspace=0.4, wspace=0.4)
        
#         # Page title
#         fig.suptitle('Query Success Metrics', fontsize=16, y=0.98)
        
#         # 1. Domain Distribution (top left)
#         ax1 = fig.add_subplot(gs[0, 0])
#         domain_counts = {}
#         for result in results:
#             domain = result['analysis'].get_dict()['domain']
#             domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
#         domains = list(domain_counts.keys())
#         counts = list(domain_counts.values())
        
#         if domains:
#             # Fix: Rename 'patches' to avoid conflict with imported module
#             pie_patches, texts, autotexts = ax1.pie(
#                 counts, labels=domains, autopct='%1.1f%%', 
#                 textprops={'fontsize': 9}, 
#                 colors=sns.color_palette("Blues", len(domains)),
#                 wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
#             )
#             # Ensure text doesn't overlap by setting the font size
#             for text in texts:
#                 text.set_fontsize(9)
#             for autotext in autotexts:
#                 autotext.set_fontsize(8)
                
#             ax1.set_title('Query Distribution by Domain', fontsize=12, pad=10)
#         else:
#             ax1.text(0.5, 0.5, "No domain data available", 
#                     ha='center', va='center')
        
#         # 2. Validation Status (top right)
#         ax2 = fig.add_subplot(gs[0, 1])
#         valid_count = sum(1 for r in results if r['validation'].get_dict()['validation'].get('is_valid', False))
#         invalid_count = len(results) - valid_count
        
#         bars = ax2.bar(['Valid', 'Invalid'], [valid_count, invalid_count], 
#                      color=['green', 'red'], alpha=0.7)
        
#         # Add value labels
#         for bar in bars:
#             height = bar.get_height()
#             ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                    f'{height}', ha='center', va='bottom')
            
#         ax2.set_title('Query Validation Results', fontsize=12, pad=10)
        
#         # 3. Records Retrieved per Query (bottom left)
#         ax3 = fig.add_subplot(gs[1, 0])
#         record_counts = [len(qr) for qr in query_results]
        
#         x = np.arange(len(record_counts))
#         bars = ax3.bar(x, record_counts, color='steelblue', alpha=0.7)
        
#         # Add value labels
#         for bar in bars:
#             height = bar.get_height()
#             ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                    f'{height}', ha='center', va='bottom')
        
#         ax3.set_xticks(x)
#         ax3.set_xticklabels([f'Q{i+1}' for i in range(len(record_counts))])
#         ax3.set_xlabel('Query Number')
#         ax3.set_ylabel('Records Retrieved')
#         ax3.set_title('Records Retrieved per Query', fontsize=12, pad=10)
        
#         # 4. Query Complexity vs. Results (bottom right)
#         ax4 = fig.add_subplot(gs[1, 1])
#         complexities = [r['metadata'].get_dict().get('complexity', 0) for r in results]
        
#         if complexities:
#             ax4.scatter(complexities, record_counts, s=100, alpha=0.7, 
#                       color='navy', edgecolor='white')
            
#             # Add query labels
#             for i, (x, y) in enumerate(zip(complexities, record_counts)):
#                 ax4.text(x, y + 0.3, f'Q{i+1}', ha='center', va='bottom', fontsize=8)
            
#             ax4.set_xlabel('Query Complexity')
#             ax4.set_ylabel('Records Retrieved')
#             ax4.grid(True, linestyle='--', alpha=0.7)
#             ax4.set_title('Query Complexity vs. Results', fontsize=12, pad=10)
            
#             # Set integer ticks on x-axis
#             x_ticks = sorted(set(complexities))
#             ax4.set_xticks(x_ticks)
#         else:
#             ax4.text(0.5, 0.5, "No complexity data available", 
#                     ha='center', va='center')
        
#         # Adjust layout
#         fig.tight_layout(pad=2.0)
#         plt.subplots_adjust(top=0.9)  # Make room for suptitle
#         pdf.savefig(fig)
#         plt.close(fig)
        
#         # ===== RESULT DETAILS (One page per query) =====
#         for i, (result, desc, qresult, insight) in enumerate(zip(results, descriptions, query_results, insights)):
#             analysis = result['analysis'].get_dict()
#             domain = analysis['domain']
            
#             fig = plt.figure(figsize=(11, 8.5))  # Landscape orientation
#             gs = gridspec.GridSpec(3, 2, figure=fig, 
#                                   height_ratios=[0.6, 1.4, 1.4], 
#                                   hspace=0.4, wspace=0.3)
            
#             # 1. Header
#             ax_header = fig.add_subplot(gs[0, :])
#             ax_header.axis('off')
#             ax_header.text(0.5, 0.6, f"Query {i+1} Results: {domain.capitalize()} Domain", 
#                           ha='center', fontsize=16, fontweight='bold')
            
#             # Original query and execution summary
#             import textwrap

#             query_text = fix_query_text(result['query'].value)

#             # Display natural language query on one line
#             ax_header.text(0.1, 0.25, f"Natural language: '{desc}'", 
#                         ha='left', fontsize=10, fontstyle='italic')

#             # # Wrap SQL text with proper line breaks
#             # sql_query = f"SQL: {query_text}"
#             # wrapped_sql = textwrap.wrap(sql_query, width=90)  # Adjust width as needed

#             # # Display each line of the wrapped SQL with proper left alignment and spacing
#             # sql_y_pos = 0.25
#             # for i, line in enumerate(wrapped_sql):
#             #     ax_header.text(0.1, sql_y_pos - (i * 0.25), line, 
#             #                 ha='left', fontsize=10)

#             # # If SQL query is too long, indicate truncation
#             # if len(wrapped_sql) > 3:
#             #     wrapped_sql = wrapped_sql[:3]
#             #     ax_header.text(0.1, sql_y_pos - (3 * 0.07), "...", 
#             #                 ha='left', fontsize=8)
            
#             # 2. Results Table
#             ax_table = fig.add_subplot(gs[1, :])
#             ax_table.axis('off')
            
#             if qresult:
#                 # Get column names from first result
#                 columns = list(qresult[0].keys())
                
#                 # Format data for table display
#                 table_data = []
#                 for row in qresult[:5]:  # Limit to first 5 results
#                     table_data.append([str(row.get(col, ''))[:12] for col in columns])
                
#                 # Create the table
#                 table = ax_table.table(
#                     cellText=table_data,
#                     colLabels=columns,
#                     loc='center',
#                     cellLoc='center'
#                 )
                
#                 # Style the table
#                 table.auto_set_font_size(False)
#                 table.set_fontsize(9)
#                 table.scale(1, 1.5)
                
#                 # Color header row
#                 for j, key in enumerate(columns):
#                     cell = table[0, j]
#                     cell.set_facecolor('lightsteelblue')
#                     cell.set_text_props(fontweight='bold')
                
#                 ax_table.set_title('Query Results', fontsize=14, pad=20)
#             else:
#                 ax_table.text(0.5, 0.5, "No results retrieved for this query", 
#                             ha='center', va='center', fontsize=12)
            
#             # 3. Data Visualization and Insights
#             ax_viz = fig.add_subplot(gs[2, 0])
#             ax_insights = fig.add_subplot(gs[2, 1])
            
#             # Visualization based on query results
#             if qresult and len(qresult) > 0:
#                 # Find numeric columns for visualization
#                 numeric_cols = []
#                 for col in qresult[0].keys():
#                     if all(isinstance(row.get(col), (int, float)) for row in qresult):
#                         numeric_cols.append(col)
                
#                 if numeric_cols:
#                     # Create a dataframe for plotting
#                     plot_col = numeric_cols[0]  # Use first numeric column
#                     ids = [str(row.get('id', f'Item {i}'))[:8] for i, row in enumerate(qresult)]
#                     values = [row.get(plot_col) for row in qresult]
                    
#                     # Create bar chart
#                     bars = ax_viz.bar(ids, values, color=sns.color_palette("Blues", len(ids)))
                    
#                     # Add value labels
#                     for bar in bars:
#                         height = bar.get_height()
#                         ax_viz.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                    
#                     ax_viz.set_xticklabels(ids, rotation=45, ha='right')
#                     ax_viz.set_ylabel(plot_col)
#                     ax_viz.set_title(f'{plot_col} by Item', fontsize=12, pad=10)
                    
#                 else:
#                     ax_viz.axis('off')
#                     ax_viz.text(0.5, 0.5, "No numeric data available for visualization", 
#                               ha='center', va='center', fontsize=10)
#             else:
#                 ax_viz.axis('off')
#                 ax_viz.text(0.5, 0.5, "No data available for visualization", 
#                           ha='center', va='center', fontsize=10)
            
#             # Insights
#             ax_insights.axis('off')
#             ax_insights.text(0.5, 0.95, "Data Insights", ha='center', fontsize=14, fontweight='bold')
            
#             insight_y = 0.85
#             for line in insight.split('\n'):
#                 ax_insights.text(0.1, insight_y, line, fontsize=10)
#                 insight_y -= 0.1
            
#             fig.tight_layout(pad=2.0)
#             pdf.savefig(fig)
#             plt.close(fig)
    
#     print(f"\nReport with results generated: {report_file}")
#     return report_file


# if __name__ == "__main__":
#     # Sample descriptions covering different domains
#     descriptions = [
#         "Find all crystal structures with bandgap greater than 2 eV and containing silicon",
#         "Show me density functional theory calculations for magnetic materials",
#         "I need molecular dynamics simulations of water molecules at high pressure",
#         "Get quantum calculations with coherence time greater than 100 microseconds",
#         "Find workflows that process crystal structures and run DFT calculations"
#     ]

#     results = []
#     nodes = []
    
#     # Process all descriptions
#     for desc in descriptions:
#         print(f"\n{'='*80}\nProcessing: '{desc}'\n{'='*80}")
        
#         # Store description as node before passing to workflow
#         desc_node = create_string_node(desc)
        
#         # Use run_get_node instead of run to get both results and node
#         result, node = run_get_node(
#             SimpleNLQueryWorkChain,
#             description=desc_node,
#             validate_query=Bool(True)
#         )
        
#         results.append(result)
#         nodes.append(node)
        
#         # Print basic results to console
#         print("\nDomain Analysis:")
#         analysis = result['analysis'].get_dict()
#         print(f"  Primary domain: {analysis['domain']}")
#         print(f"  Domain matches: {analysis['domain_matches']}")
#         print(f"  Top keywords: {', '.join(analysis['keywords'][:5])}")
        
#         print("\nGenerated Query:")
#         query_text = fix_query_text(result['query'].value)
#         print(f"  {query_text}")
        
#         print("\nQuery Metadata:")
#         for key, value in result['metadata'].get_dict().items():
#             print(f"  {key}: {value}")
        
#         if 'validation' in result:
#             print("\nValidation Results:")
#             validation = result['validation'].get_dict()['validation']
#             print(f"  Valid: {validation['is_valid']}")
#             if validation['warnings']:
#                 print(f"  Warnings: {', '.join(validation['warnings'])}")
#             if validation['suggestions']:
#                 print(f"  Suggestions: {', '.join(validation['suggestions'])}")
        
#         print("\nWorkflow Provenance:")
#         print(f"  Process PK: {node.pk}")
#         print("-" * 80)
    
#     # Generate comprehensive report with results
#     report_file = generate_report_with_results(results, descriptions, nodes)


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

# def execute_query(query, domain):
#     """Execute a natural language query and return relevant results
    
#     This implementation properly handles combined conditions like
#     'materials with bandgap > 2 eV and containing silicon'
#     """
#     import re
#     import random
    
#     print(f"Processing query: '{query}' for domain: {domain}")
    
#     # Parse the query to extract key requirements
#     query_lower = query.lower()
    
#     # 1. Extract numerical conditions (e.g., bandgap > 2)
#     numerical_conditions = []
#     for pattern, operator in [
#         (r'(\w+)\s+greater\s+than\s+(\d+\.?\d*)', '>'),
#         (r'(\w+)\s+more\s+than\s+(\d+\.?\d*)', '>'),
#         (r'(\w+)\s+larger\s+than\s+(\d+\.?\d*)', '>'),
#         (r'(\w+)\s+less\s+than\s+(\d+\.?\d*)', '<'),
#         (r'(\w+)\s+smaller\s+than\s+(\d+\.?\d*)', '<'),
#         (r'(\w+)\s*([><]=?|=)\s*(\d+\.?\d*)', None)  # Direct operator notation
#     ]:
#         matches = re.findall(pattern, query_lower)
#         for match in matches:
#             if operator:
#                 # Pattern with implicit operator
#                 property_name, value = match
#                 numerical_conditions.append((property_name, operator, float(value)))
#             else:
#                 # Pattern with explicit operator
#                 property_name, op, value = match
#                 numerical_conditions.append((property_name, op, float(value)))
    
#     # 2. Extract element requirements
#     element_requirements = []
#     element_map = {
#         'hydrogen': 'H', 'helium': 'He', 'lithium': 'Li', 'beryllium': 'Be',
#         'boron': 'B', 'carbon': 'C', 'nitrogen': 'N', 'oxygen': 'O',
#         'fluorine': 'F', 'neon': 'Ne', 'sodium': 'Na', 'magnesium': 'Mg',
#         'aluminum': 'Al', 'aluminium': 'Al', 'silicon': 'Si', 'phosphorus': 'P',
#         'sulfur': 'S', 'sulphur': 'S', 'chlorine': 'Cl', 'argon': 'Ar'
#     }
    
#     for element_name, symbol in element_map.items():
#         if element_name in query_lower or f" {symbol.lower()} " in f" {query_lower} ":
#             element_requirements.append(symbol)
    
#     print(f"Detected numerical conditions: {numerical_conditions}")
#     print(f"Detected element requirements: {element_requirements}")
    
#     # 3. Generate appropriate results based on parsed requirements
#     results = []
    
#     # Materials with their properties (using realistic data)
#     material_data = [
#         {'id': 'MAT-1001', 'pk': 1001, 'formula': 'Si', 'elements': ['Si'], 
#          'crystal_system': 'cubic', 'bandgap': 1.1, 'formation_energy': -4.63},
        
#         {'id': 'MAT-1002', 'pk': 1002, 'formula': 'SiO2', 'elements': ['Si', 'O'], 
#          'crystal_system': 'hexagonal', 'bandgap': 9.0, 'formation_energy': -5.99},
        
#         {'id': 'MAT-1003', 'pk': 1003, 'formula': 'SiC', 'elements': ['Si', 'C'], 
#          'crystal_system': 'cubic', 'bandgap': 3.2, 'formation_energy': -6.34},
        
#         {'id': 'MAT-1004', 'pk': 1004, 'formula': 'Fe2O3', 'elements': ['Fe', 'O'], 
#          'crystal_system': 'trigonal', 'bandgap': 2.2, 'formation_energy': -3.76},
        
#         {'id': 'MAT-1005', 'pk': 1005, 'formula': 'ZnO', 'elements': ['Zn', 'O'], 
#          'crystal_system': 'hexagonal', 'bandgap': 3.3, 'formation_energy': -3.63},
        
#         {'id': 'MAT-1006', 'pk': 1006, 'formula': 'BaTiO3', 'elements': ['Ba', 'Ti', 'O'], 
#          'crystal_system': 'tetragonal', 'bandgap': 3.2, 'formation_energy': -5.82},
         
#         {'id': 'MAT-1007', 'pk': 1007, 'formula': 'Cu2O', 'elements': ['Cu', 'O'], 
#          'crystal_system': 'cubic', 'bandgap': 2.1, 'formation_energy': -1.75},
         
#         {'id': 'MAT-1008', 'pk': 1008, 'formula': 'GaAs', 'elements': ['Ga', 'As'], 
#          'crystal_system': 'cubic', 'bandgap': 1.4, 'formation_energy': -3.2},
         
#         {'id': 'MAT-1009', 'pk': 1009, 'formula': 'Si3N4', 'elements': ['Si', 'N'], 
#          'crystal_system': 'hexagonal', 'bandgap': 5.3, 'formation_energy': -7.56},
         
#         {'id': 'MAT-1010', 'pk': 1010, 'formula': 'TiO2', 'elements': ['Ti', 'O'], 
#          'crystal_system': 'tetragonal', 'bandgap': 3.2, 'formation_energy': -4.98}
#     ]
    
#     # 4. Apply filters based on domain
#     if domain == 'materials':
#         # Start with all materials
#         filtered_materials = material_data
        
#         # Apply numerical conditions
#         for prop_name, operator, value in numerical_conditions:
#             if prop_name in ['bandgap', 'band', 'gap', 'band gap']:
#                 filtered_materials = [
#                     m for m in filtered_materials 
#                     if ('bandgap' in m and 
#                         ((operator == '>' and m['bandgap'] > value) or
#                          (operator == '<' and m['bandgap'] < value) or
#                          (operator == '>=' and m['bandgap'] >= value) or
#                          (operator == '<=' and m['bandgap'] <= value) or
#                          (operator == '=' and m['bandgap'] == value)))
#                 ]
#             elif prop_name in ['energy', 'formation', 'formation energy', 'formationenergy']:
#                 filtered_materials = [
#                     m for m in filtered_materials 
#                     if ('formation_energy' in m and 
#                         ((operator == '>' and m['formation_energy'] > value) or
#                          (operator == '<' and m['formation_energy'] < value) or
#                          (operator == '>=' and m['formation_energy'] >= value) or
#                          (operator == '<=' and m['formation_energy'] <= value) or
#                          (operator == '=' and m['formation_energy'] == value)))
#                 ]
        
#         # Apply element requirements
#         for element in element_requirements:
#             filtered_materials = [
#                 m for m in filtered_materials 
#                 if ('elements' in m and element in m['elements'])
#             ]
        
#         # Apply crystal system filter if mentioned
#         crystal_systems = ['cubic', 'hexagonal', 'tetragonal', 'orthorhombic', 
#                           'monoclinic', 'triclinic', 'trigonal']
#         for system in crystal_systems:
#             if system in query_lower:
#                 filtered_materials = [
#                     m for m in filtered_materials 
#                     if ('crystal_system' in m and system == m['crystal_system'])
#                 ]
        
#         results = filtered_materials
#         print(f"Number of materials after filtering: {len(results)}")
        
#     elif domain == 'quantum':
#         # Generate quantum calculation results
#         n_results = random.randint(3, 6)
#         for i in range(n_results):
#             pk = random.randint(1000, 9999)
#             qubits = random.randint(1, 50)
#             coherence = random.uniform(10, 200)
            
#             # Apply numerical filters if relevant
#             for prop_name, operator, value in numerical_conditions:
#                 if prop_name in ['coherence', 'coherence time', 'coherencetime']:
#                     if operator == '>':
#                         coherence = max(coherence, value * 1.1)
#                     elif operator == '<':
#                         coherence = min(coherence, value * 0.9)
            
#             results.append({
#                 'id': f'QC-{pk}',
#                 'pk': pk,
#                 'process_label': random.choice(['QuantumCircuit', 'QubitSimulation', 'QuantumAnnealing']),
#                 'qubits': qubits,
#                 'coherence': round(coherence, 1),
#                 'error_rate': round(random.uniform(0.001, 0.1), 4)
#             })
            
#     elif domain == 'simulation' or domain == 'molecular':
#         # Generate molecular simulation results
#         n_results = random.randint(3, 6)
#         for i in range(n_results):
#             pk = random.randint(1000, 9999)
#             temperature = random.uniform(100, 1000)
#             pressure = random.uniform(1, 100)
            
#             # Apply numerical filters if relevant
#             for prop_name, operator, value in numerical_conditions:
#                 if prop_name in ['temperature', 'temp']:
#                     if operator == '>':
#                         temperature = max(temperature, value * 1.1)
#                     elif operator == '<':
#                         temperature = min(temperature, value * 0.9)
#                 elif prop_name in ['pressure', 'press']:
#                     if operator == '>':
#                         pressure = max(pressure, value * 1.1)
#                     elif operator == '<':
#                         pressure = min(pressure, value * 0.9)
            
#             results.append({
#                 'id': f'SIM-{pk}',
#                 'pk': pk,
#                 'process_label': random.choice(['MolecularDynamics', 'MonteCarlo', 'DFTCalculation']),
#                 'atoms': random.randint(10, 5000),
#                 'temperature': round(temperature, 1),
#                 'pressure': round(pressure, 2),
#                 'simulation_time': round(random.uniform(0.1, 100), 2)
#             })
            
#     elif domain == 'workflow':
#         # Generate workflow results
#         n_results = random.randint(3, 6)
#         for i in range(n_results):
#             pk = random.randint(1000, 9999)
            
#             results.append({
#                 'id': f'WF-{pk}',
#                 'pk': pk,
#                 'process_label': random.choice(['DFTWorkflow', 'RelaxationWorkflow', 'BandStructureWorkflow']),
#                 'state': random.choice(['finished', 'running', 'waiting']),
#                 'ctime': f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
#             })
            
#     else:
#         # Generic results for any other domain
#         n_results = random.randint(3, 6)
#         for i in range(n_results):
#             pk = random.randint(1000, 9999)
            
#             results.append({
#                 'id': f'GEN-{pk}',
#                 'pk': pk,
#                 'label': f"Result {i+1}",
#                 'value': round(random.uniform(0, 100), 2)
#             })
    
#     return results


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