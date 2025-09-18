import json
import os
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from dotenv import load_dotenv

# Load environment variables
load_dotenv('api_keys.env')

class CombinedReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        
        # Set up matplotlib style for professional charts
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Create charts directory if it doesn't exist
        if not os.path.exists('charts'):
            os.makedirs('charts')
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Executive title style
        self.styles.add(ParagraphStyle(
            'ExecutiveTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E4057'),
            fontName='Helvetica-Bold'
        ))
        
        # Executive subtitle style
        self.styles.add(ParagraphStyle(
            'ExecutiveSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#2E4057'),
            fontName='Helvetica-Bold'
        ))
        
        # Executive body style
        self.styles.add(ParagraphStyle(
            'ExecutiveBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            alignment=TA_LEFT,
            fontName='Helvetica'
        ))
        
        # Executive summary text style
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummaryText',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            spaceBefore=15,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            leading=20,  # Increased line spacing
            leftIndent=0,
            rightIndent=0,
            leftPadding=0,
            rightPadding=0
        ))
    
    def load_json_data(self, json_file_path):
        """Load JSON data from file"""
        try:
            with open(json_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ JSON file not found: {json_file_path}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON format in file: {json_file_path}")
            return None
    
    def call_perplexity_api(self, prompt, max_tokens=500):
        """Call Perplexity API to generate content"""
        try:
            headers = {
                'Authorization': f'Bearer {self.perplexity_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'sonar',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a business analyst expert specializing in executive summaries for customer portfolio reports. Generate concise, professional, and actionable content suitable for C-level executives.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': max_tokens,
                'temperature': 0.7
            }
            
            response = requests.post(
                'https://api.perplexity.ai/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"❌ Perplexity API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling Perplexity API: {str(e)}")
            return None
    
    def generate_combined_executive_summary_with_api(self, all_customers_data):
        """Generate combined executive summary using Perplexity API"""
        try:
            # Prepare comprehensive data for the prompt
            total_customers = len(all_customers_data)
            total_stores = sum(customer.get('CustomerClassification', {}).get('NumberOfStores', 0) for customer in all_customers_data)
            total_volume = sum(customer.get('CustomerClassification', {}).get('TotalQuantitySold', 0) for customer in all_customers_data)
            
            total_accepted = 0
            total_rejected = 0
            total_purchased = 0
            total_up_sell = 0
            
            customer_summaries = []
            for customer in all_customers_data:
                customer_info = customer.get('CustomerInfo', {})
                customer_classification = customer.get('CustomerClassification', {})
                accepted_recommendations = customer.get('AcceptedRecommendations', [])
                rejected_recommendations = customer.get('RejectedRecommendations', [])
                already_purchased_recommendations = customer.get('AlreadyPurchasedRecommendations', [])
                
                customer_name = customer_info.get('CustomerName', 'Unknown')
                customer_id = customer_info.get('CustomerID', 'Unknown')
                stores = customer_classification.get('NumberOfStores', 0)
                volume = customer_classification.get('TotalQuantitySold', 0)
                customer_type = customer_classification.get('CustomerType', 'Unknown')
                
                accepted_count = sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
                rejected_count = sum(len(rec.get('RejectedCrossSell', [])) for rec in rejected_recommendations)
                purchased_count = sum(len(rec.get('AlreadyPurchasedCrossSell', [])) for rec in already_purchased_recommendations)
                
                total_accepted += accepted_count
                total_rejected += rejected_count
                total_purchased += purchased_count
                
                for rec in accepted_recommendations:
                    total_up_sell += len(rec.get('UpSell', []))
                
                customer_summaries.append(f"- {customer_name} ({customer_id}): {stores} stores, {volume:,} units, {accepted_count} accepted opportunities, {customer_type}")
            
            prompt = f"""
            Generate a comprehensive executive summary for a customer portfolio analysis report with the following data:
            
            Portfolio Overview:
            - Total Customers: {total_customers}
            - Total Stores: {total_stores:,}
            - Total Purchase Volume: {total_volume:,} units
            
            Analysis Results:
            - Accepted Cross-Sell Opportunities: {total_accepted} items
            - Rejected Opportunities: {total_rejected} items (quality control)
            - Already Purchased: {total_purchased} items (duplicate prevention)
            - Up-Sell Opportunities: {total_up_sell} items
            
            Customer Breakdown:
            {chr(10).join(customer_summaries)}
            
            Generate a very concise executive summary (1-2 short paragraphs, maximum 200 words) that:
            1. Provides portfolio-level insights
            2. Highlights key performance metrics
            3. Identifies strategic opportunities
            4. Concludes with business impact
            
            Use simple, direct language. Be extremely concise. Target busy C-level executives.
            Focus on the most critical portfolio insights only.
            """
            
            api_result = self.call_perplexity_api(prompt, max_tokens=400)
            
            if api_result:
                return api_result
            else:
                # Fallback summary
                return f"Portfolio analysis of {total_customers} customers reveals {total_accepted} validated cross-sell opportunities across {total_stores:,} stores with {total_volume:,} units purchase volume. Key opportunities span multiple customer segments with strong potential for revenue growth and market expansion."
                
        except Exception as e:
            print(f"❌ Error generating combined executive summary: {str(e)}")
            return "Portfolio analysis completed with comprehensive customer insights and strategic recommendations."
    
    
    
    def cleanup_chart_files(self):
        """Clean up temporary chart files"""
        try:
            if os.path.exists('charts'):
                for file in os.listdir('charts'):
                    if file.endswith('.png'):
                        os.remove(os.path.join('charts', file))
                print("🧹 Cleaned up temporary chart files")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up chart files: {str(e)}")
    
    def create_portfolio_overview_chart(self, all_customers_data):
        """Create portfolio overview pie chart"""
        try:
            # Calculate metrics
            total_accepted = 0
            total_rejected = 0
            total_purchased = 0
            total_up_sell = 0
            
            for customer in all_customers_data:
                accepted_recommendations = customer.get('AcceptedRecommendations', [])
                rejected_recommendations = customer.get('RejectedRecommendations', [])
                already_purchased_recommendations = customer.get('AlreadyPurchasedRecommendations', [])
                
                total_accepted += sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
                total_rejected += sum(len(rec.get('RejectedCrossSell', [])) for rec in rejected_recommendations)
                total_purchased += sum(len(rec.get('AlreadyPurchasedCrossSell', [])) for rec in already_purchased_recommendations)
                
                for rec in accepted_recommendations:
                    total_up_sell += len(rec.get('UpSell', []))
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            labels = ['Accepted\nOpportunities', 'Rejected\nOpportunities', 'Already\nPurchased', 'Up-Sell\nOpportunities']
            sizes = [total_accepted, total_rejected, total_purchased, total_up_sell]
            colors = ['#2E8B57', '#FF6B6B', '#FFD93D', '#6BCF7F']
            
            # Only show non-zero values
            non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
            if non_zero_data:
                labels, sizes, colors = zip(*non_zero_data)
                
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                                startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
                
                # Enhance text appearance
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(14)
                    autotext.set_weight('bold')
                
                ax.set_title('Portfolio Opportunities Distribution', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = 'charts/portfolio_overview.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            print(f"❌ Error creating portfolio overview chart: {str(e)}")
            return None
    
    
    
    def create_kpi_dashboard_chart(self, all_customers_data):
        """Create KPI dashboard with key metrics"""
        try:
            # Calculate KPIs
            total_customers = len(all_customers_data)
            total_stores = sum(customer.get('CustomerClassification', {}).get('NumberOfStores', 0) for customer in all_customers_data)
            total_volume = sum(customer.get('CustomerClassification', {}).get('TotalQuantitySold', 0) for customer in all_customers_data)
            
            total_accepted = 0
            total_rejected = 0
            total_purchased = 0
            total_up_sell = 0
            
            for customer in all_customers_data:
                accepted_recommendations = customer.get('AcceptedRecommendations', [])
                rejected_recommendations = customer.get('RejectedRecommendations', [])
                already_purchased_recommendations = customer.get('AlreadyPurchasedRecommendations', [])
                
                total_accepted += sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
                total_rejected += sum(len(rec.get('RejectedCrossSell', [])) for rec in rejected_recommendations)
                total_purchased += sum(len(rec.get('AlreadyPurchasedCrossSell', [])) for rec in already_purchased_recommendations)
                
                for rec in accepted_recommendations:
                    total_up_sell += len(rec.get('UpSell', []))
            
            # Calculate success rate
            total_opportunities = total_accepted + total_rejected
            success_rate = (total_accepted / total_opportunities * 100) if total_opportunities > 0 else 0
            
            # Create KPI dashboard
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # KPI 1: Total Opportunities
            ax1.pie([total_accepted, total_rejected, total_purchased], 
                   labels=['Accepted', 'Rejected', 'Already Purchased'],
                   colors=['#2E8B57', '#FF6B6B', '#FFD93D'],
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title('Opportunity Distribution', fontsize=14, fontweight='bold')
            
            # KPI 2: Success Rate Gauge
            ax2.text(0.5, 0.5, f'{success_rate:.1f}%', ha='center', va='center', 
                    fontsize=24, fontweight='bold', transform=ax2.transAxes)
            ax2.text(0.5, 0.3, 'Success Rate', ha='center', va='center', 
                    fontsize=12, transform=ax2.transAxes)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            
            # Add circular progress indicator
            circle = plt.Circle((0.5, 0.5), 0.3, fill=False, linewidth=8, 
                              color='#4A90E2' if success_rate >= 70 else '#FF6B6B' if success_rate < 50 else '#FFD93D')
            ax2.add_patch(circle)
            
            # KPI 3: Portfolio Scale
            metrics = ['Customers', 'Stores', 'Volume (K units)']
            values = [total_customers, total_stores, total_volume/1000]
            bars = ax3.bar(metrics, values, color=['#2E8B57', '#4A90E2', '#FF6B6B'], alpha=0.8)
            ax3.set_title('Portfolio Scale', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{height:,.0f}', ha='center', va='bottom', fontweight='bold')
            
            # KPI 4: Revenue Potential
            revenue_estimate = total_accepted * 1000  # $1000 per opportunity
            ax4.text(0.5, 0.6, f'${revenue_estimate:,.0f}', ha='center', va='center', 
                    fontsize=20, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.5, 0.4, 'Estimated Revenue\nPotential', ha='center', va='center', 
                    fontsize=12, transform=ax4.transAxes)
            ax4.text(0.5, 0.2, f'Based on {total_accepted} opportunities', ha='center', va='center', 
                    fontsize=10, transform=ax4.transAxes)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            
            plt.suptitle('Portfolio KPI Dashboard', fontsize=16, fontweight='bold', y=0.95)
            plt.tight_layout()
            
            # Save chart
            chart_path = 'charts/kpi_dashboard.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            print(f"❌ Error creating KPI dashboard: {str(e)}")
            return None
    
    def generate_combined_executive_summary(self, all_customers_data):
        """Generate executive summary for all customers combined"""
        story = []
        
        # Main title
        story.append(Paragraph("Customer Portfolio Executive Report", self.styles['ExecutiveTitle']))
        story.append(Spacer(1, 20))
        
        # Portfolio overview
        story.append(Paragraph("Portfolio Overview", self.styles['ExecutiveSubtitle']))
        
        total_customers = len(all_customers_data)
        total_stores = sum(customer.get('CustomerClassification', {}).get('NumberOfStores', 0) for customer in all_customers_data)
        total_volume = sum(customer.get('CustomerClassification', {}).get('TotalQuantitySold', 0) for customer in all_customers_data)
        
        # Calculate total opportunities
        total_accepted = 0
        total_rejected = 0
        total_purchased = 0
        total_up_sell = 0
        
        for customer in all_customers_data:
            accepted_recommendations = customer.get('AcceptedRecommendations', [])
            rejected_recommendations = customer.get('RejectedRecommendations', [])
            already_purchased_recommendations = customer.get('AlreadyPurchasedRecommendations', [])
            
            total_accepted += sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
            total_rejected += sum(len(rec.get('RejectedCrossSell', [])) for rec in rejected_recommendations)
            total_purchased += sum(len(rec.get('AlreadyPurchasedCrossSell', [])) for rec in already_purchased_recommendations)
            
            for rec in accepted_recommendations:
                total_up_sell += len(rec.get('UpSell', []))
        
        # Portfolio summary table
        portfolio_data = [
            ['Metric', 'Value'],
            ['Total Customers', f"{total_customers}"],
            ['Total Stores', f"{total_stores:,}"],
            ['Total Purchase Volume', f"{total_volume:,} units"],
            ['Accepted Opportunities', f"{total_accepted} items"],
            ['Rejected Opportunities', f"{total_rejected} items"],
            ['Already Purchased', f"{total_purchased} items"],
            ['Up-Sell Opportunities', f"{total_up_sell} items"]
        ]
        
        portfolio_table = Table(portfolio_data, colWidths=[2.5*inch, 2.5*inch])
        portfolio_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E4057')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#2E4057')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
        ]))
        
        story.append(portfolio_table)
        story.append(Spacer(1, 20))
        
        # Customer performance summary
        story.append(Paragraph("Customer Performance Summary", self.styles['ExecutiveSubtitle']))
        
        # Create customer comparison table
        customer_data = [['Customer', 'Stores', 'Volume', 'Accepted', 'Classification', 'Priority']]
        
        for customer in all_customers_data:
            customer_info = customer.get('CustomerInfo', {})
            customer_classification = customer.get('CustomerClassification', {})
            accepted_recommendations = customer.get('AcceptedRecommendations', [])
            
            customer_name = customer_info.get('CustomerName', 'Unknown')
            customer_id = customer_info.get('CustomerID', 'Unknown')
            stores = customer_classification.get('NumberOfStores', 0)
            volume = customer_classification.get('TotalQuantitySold', 0)
            customer_type = customer_classification.get('CustomerType', 'Unknown')
            
            total_accepted_customer = sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
            
            # Determine priority based on opportunities and classification
            if customer_type == 'CHG Own Sales Customer' and total_accepted_customer > 5:
                priority = "High"
            elif total_accepted_customer > 3:
                priority = "Medium"
            else:
                priority = "Low"
            
            customer_data.append([
                f"{customer_name} ({customer_id})",
                f"{stores:,}",
                f"{volume:,}",
                f"{total_accepted_customer}",
                customer_type,
                priority
            ])
        
        customer_table = Table(customer_data, colWidths=[1.8*inch, 0.9*inch, 1.1*inch, 0.9*inch, 1.4*inch, 0.8*inch])
        customer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (2, -1), 'CENTER'),
            ('ALIGN', (3, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#4A90E2')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F8FF')])
        ]))
        
        story.append(customer_table)
        story.append(Spacer(1, 20))
        
        # Generate KPI Dashboard Chart
        print("📊 Creating KPI Dashboard Chart...")
        kpi_chart_path = self.create_kpi_dashboard_chart(all_customers_data)
        if kpi_chart_path and os.path.exists(kpi_chart_path):
            story.append(Paragraph("Portfolio KPI Dashboard", self.styles['ExecutiveSubtitle']))
            story.append(Spacer(1, 15))
            kpi_img = Image(kpi_chart_path, width=6*inch, height=5*inch)
            story.append(kpi_img)
            story.append(Spacer(1, 20))
        
        # Generate Portfolio Overview Chart
        print("📊 Creating Portfolio Overview Chart...")
        portfolio_chart_path = self.create_portfolio_overview_chart(all_customers_data)
        if portfolio_chart_path and os.path.exists(portfolio_chart_path):
            story.append(Paragraph("Portfolio Opportunities Distribution", self.styles['ExecutiveSubtitle']))
            story.append(Spacer(1, 15))
            portfolio_img = Image(portfolio_chart_path, width=6*inch, height=4.8*inch)
            story.append(portfolio_img)
            story.append(Spacer(1, 20))
        
        # Generate combined executive summary using Perplexity API
        print("🤖 Generating Combined Executive Summary with Perplexity API...")
        combined_summary = self.generate_combined_executive_summary_with_api(all_customers_data)
        
        story.append(Paragraph("Executive Summary", self.styles['ExecutiveSubtitle']))
        story.append(Spacer(1, 15))
        story.append(Paragraph(combined_summary, self.styles['ExecutiveSummaryText']))
        
        return story
    
    
    
    def generate_combined_report(self, json_files, output_pdf_path):
        """Generate a combined CEO/CXO-friendly report from multiple JSON files (first page only)"""
        try:
            doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, topMargin=1*inch)
            story = []
            
            # Load all customer data
            all_customers_data = []
            for json_file in json_files:
                json_file_path = os.path.join('recommendations', json_file)
                print(f"📄 Processing: {json_file_path}")
                data = self.load_json_data(json_file_path)
                if data:
                    all_customers_data.append(data)
            
            if not all_customers_data:
                print("❌ No valid customer data found")
                return False
            
            # Generate combined executive summary (first page only)
            story.extend(self.generate_combined_executive_summary(all_customers_data))
            
            # Build PDF
            doc.build(story)
            print(f"✅ Combined CEO/CXO PDF report generated successfully (first page only): {output_pdf_path}")
            
            # Clean up temporary chart files
            self.cleanup_chart_files()
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating combined report: {str(e)}")
            return False

def main():
    """Main function to generate combined PDF report"""
    generator = CombinedReportGenerator()
    
    # Find all JSON files in the recommendations folder
    json_files = [f for f in os.listdir('recommendations') if f.startswith('recommendations_') and f.endswith('.json')]
    
    if not json_files:
        print("❌ No recommendation JSON files found in current directory")
        return
    
    # Sort by modification time to get the most recent
    json_files.sort(key=lambda x: os.path.getmtime(os.path.join('recommendations', x)), reverse=True)
    
    print(f"📄 Found {len(json_files)} JSON files")
    
    # Generate combined PDF report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"reports/combined_portfolio_report_{timestamp}.pdf"
    
    success = generator.generate_combined_report(json_files, output_path)
    
    if success:
        print("🎉 Combined portfolio PDF report generated successfully!")
    else:
        print("❌ Failed to generate combined PDF report")

if __name__ == "__main__":
    main()
