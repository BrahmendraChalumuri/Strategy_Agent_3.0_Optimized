import json
import os
import requests
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from dotenv import load_dotenv

# Load environment variables
load_dotenv('api_keys.env')

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkgreen
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkred
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Highlight style for important information
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.darkblue,
            alignment=TA_JUSTIFY
        ))
        
        # Executive Summary styles
        self.styles.add(ParagraphStyle(
            name='ExecutiveTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='ExecutiveSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=15,
            spaceBefore=15,
            alignment=TA_CENTER,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='ExecutiveBody',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            name='ExecutiveHighlight',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=10,
            textColor=colors.darkblue,
            alignment=TA_JUSTIFY,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='ExecutiveTableHeader',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_CENTER,
            textColor=colors.white,
            fontName='Helvetica-Bold'
        ))
        
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
    
    def call_openai_api(self, prompt, max_tokens=500):
        """Call OpenAI GPT API to generate content"""
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'gpt-4',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a business analyst expert specializing in executive summaries for customer recommendation reports. Generate concise, professional, and actionable content suitable for C-level executives.'
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
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"❌ OpenAI API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {str(e)}")
            return None
    
    def generate_executive_summary_with_api(self, data):
        """Generate Executive Summary using OpenAI GPT API"""
        try:
            # Extract key data for the prompt
            customer_info = data.get('CustomerInfo', {})
            customer_classification = data.get('CustomerClassification', {})
            accepted_recommendations = data.get('AcceptedRecommendations', [])
            rejected_recommendations = data.get('RejectedRecommendations', [])
            already_purchased_recommendations = data.get('AlreadyPurchasedRecommendations', [])
            
            customer_name = customer_info.get('CustomerName', 'Unknown Customer')
            customer_id = customer_info.get('CustomerID', 'Unknown ID')
            number_of_stores = customer_classification.get('NumberOfStores', 0)
            total_quantity = customer_classification.get('TotalQuantitySold', 0)
            customer_type = customer_classification.get('CustomerType', 'Unknown')
            
            total_accepted = sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
            total_rejected = sum(len(rec.get('RejectedCrossSell', [])) for rec in rejected_recommendations)
            total_already_purchased = sum(len(rec.get('AlreadyPurchasedCrossSell', [])) for rec in already_purchased_recommendations)
            
            # Count up-sell opportunities
            total_up_sell = 0
            for rec in accepted_recommendations:
                total_up_sell += len(rec.get('UpSell', []))
            
            # Create detailed prompt for Perplexity API
            prompt = f"""
            Generate a professional executive summary for a customer recommendation analysis report with the following data:
            
            Customer Information:
            - Customer: {customer_name} ({customer_id})
            - Stores: {number_of_stores:,} stores
            - Purchase Volume: {total_quantity:,} units
            - Classification: {customer_type}
            
            Analysis Results:
            - Accepted Cross-Sell Opportunities: {total_accepted} items
            - Rejected Opportunities: {total_rejected} items (quality control)
            - Already Purchased: {total_already_purchased} items (duplicate prevention)
            - Up-Sell Opportunities: {total_up_sell} items
            
            Accepted Cross-Sell Details:
            """
            
            # Add details about accepted recommendations
            for rec in accepted_recommendations:
                item_name = rec.get('ProductName', 'Unknown')
                cross_sell_items = rec.get('CrossSell', [])
                if cross_sell_items:
                    prompt += f"- {item_name}: {len(cross_sell_items)} cross-sell opportunities\n"
                    for cross_sell in cross_sell_items:
                        product_name = cross_sell.get('SuggestedProduct', 'Unknown')
                        category = cross_sell.get('Category', 'Unknown')
                        price = cross_sell.get('Price', 0)
                        prompt += f"  * {product_name} ({category}) - ${price}\n"
            
            prompt += f"""
            
            Generate a very concise executive summary (1-2 short paragraphs, maximum 150 words) that:
            1. Briefly states customer profile and classification
            2. Highlights the key opportunity numbers (accepted/rejected/purchased)
            3. Provides one key business insight
            4. Concludes with strategic value
            
            Use simple, direct language. Be extremely concise. Target busy C-level executives.
            Focus on the most critical information only. Avoid detailed explanations.
            """
            
            # Call OpenAI API
            api_result = self.call_openai_api(prompt, max_tokens=300)
            
            if api_result:
                return api_result
            else:
                # Fallback to static content if API fails
                return self.generate_fallback_executive_summary(data)
                
        except Exception as e:
            print(f"❌ Error generating executive summary with API: {str(e)}")
            return self.generate_fallback_executive_summary(data)
    
    def generate_next_steps_with_api(self, data):
        """Generate Next Steps using Perplexity API"""
        try:
            # Extract key data for the prompt
            accepted_recommendations = data.get('AcceptedRecommendations', [])
            rejected_recommendations = data.get('RejectedRecommendations', [])
            already_purchased_recommendations = data.get('AlreadyPurchasedRecommendations', [])
            
            total_accepted = sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
            total_rejected = sum(len(rec.get('RejectedCrossSell', [])) for rec in rejected_recommendations)
            total_already_purchased = sum(len(rec.get('AlreadyPurchasedCrossSell', [])) for rec in already_purchased_recommendations)
            
            # Count up-sell opportunities
            total_up_sell = 0
            for rec in accepted_recommendations:
                total_up_sell += len(rec.get('UpSell', []))
            
            # Create detailed prompt for Perplexity API
            prompt = f"""
            Generate very concise next steps for a customer recommendation analysis report.
            
            Analysis Results:
            - Accepted Cross-Sell Opportunities: {total_accepted} items
            - Rejected Opportunities: {total_rejected} items
            - Already Purchased: {total_already_purchased} items
            - Up-Sell Opportunities: {total_up_sell} items
            
            Create 4-5 very brief next steps that are:
            - Extremely concise (maximum 15 words each)
            - Executive-appropriate
            - Action-oriented
            - Clear and direct
            
            Format as simple bullet points starting with action verbs.
            Focus only on the most critical actions. Be very brief.
            """
            
            # Call OpenAI API
            api_result = self.call_openai_api(prompt, max_tokens=250)
            
            if api_result:
                return api_result
            else:
                # Fallback to static content if API fails
                return self.generate_fallback_next_steps(data)
                
        except Exception as e:
            print(f"❌ Error generating next steps with API: {str(e)}")
            return self.generate_fallback_next_steps(data)
    
    def generate_fallback_executive_summary(self, data):
        """Fallback executive summary if API fails"""
        customer_info = data.get('CustomerInfo', {})
        customer_classification = data.get('CustomerClassification', {})
        accepted_recommendations = data.get('AcceptedRecommendations', [])
        rejected_recommendations = data.get('RejectedRecommendations', [])
        already_purchased_recommendations = data.get('AlreadyPurchasedRecommendations', [])
        
        customer_name = customer_info.get('CustomerName', 'Unknown Customer')
        customer_id = customer_info.get('CustomerID', 'Unknown ID')
        number_of_stores = customer_classification.get('NumberOfStores', 0)
        total_quantity = customer_classification.get('TotalQuantitySold', 0)
        customer_type = customer_classification.get('CustomerType', 'Unknown')
        
        total_accepted = sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
        total_rejected = sum(len(rec.get('RejectedCrossSell', [])) for rec in rejected_recommendations)
        total_already_purchased = sum(len(rec.get('AlreadyPurchasedCrossSell', [])) for rec in already_purchased_recommendations)
        
        total_up_sell = 0
        for rec in accepted_recommendations:
            total_up_sell += len(rec.get('UpSell', []))
        
        potential_level = 'strong' if total_accepted > 5 else 'moderate' if total_accepted > 0 else 'limited'
        customer_value = 'high-value' if customer_type == 'CHG Own Sales Customer' else 'medium-value' if customer_type == 'Distributor Customer' else 'developing'
        opportunity_type = 'immediate' if total_accepted > 0 else 'limited'
        
        summary_text = f"Analysis of {customer_name} ({customer_id}), with {number_of_stores:,} stores and {total_quantity:,} units purchase volume, classified as a {customer_type} with {potential_level} cross-selling potential."
        
        if total_accepted > 0:
            summary_text += f" Identification of {total_accepted} validated cross-sell opportunities across categories like biscuits, gravies, desserts, baking mixes, and pizza dough, with high probability for adoption and revenue growth."
        
        if total_rejected > 0:
            summary_text += f" Explanation for {total_rejected} rejected potential cross-sell items (compatibility checks, quality control)."
        
        if total_already_purchased > 0:
            summary_text += f" {total_already_purchased} product flagged as already purchased (preventing duplication)."
        
        if total_up_sell == 0:
            summary_text += " No up-sell opportunities detected, suggesting optimal purchase volumes."
        else:
            summary_text += f" {total_up_sell} up-sell opportunities identified for revenue enhancement."
        
        summary_text += f" Overall, the report highlights {customer_name} as a {customer_value} customer with {opportunity_type} opportunities to expand its product portfolio."
        summary_text += f" The recommended next steps are to focus on the {total_accepted} accepted cross-sell items, monitor adoption, and re-run the analysis quarterly for new opportunities."
        
        return summary_text
    
    def generate_fallback_next_steps(self, data):
        """Fallback next steps if API fails"""
        accepted_recommendations = data.get('AcceptedRecommendations', [])
        total_accepted = sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
        
        total_up_sell = 0
        for rec in accepted_recommendations:
            total_up_sell += len(rec.get('UpSell', []))
        
        steps = []
        if total_accepted > 0:
            steps.append(f"→ Prioritize execution of {total_accepted} cross-sell recommendations.")
        if total_up_sell > 0:
            steps.append(f"→ Implement {total_up_sell} up-sell opportunities.")
        steps.append("→ Monitor customer adoption and impact.")
        steps.append("→ Re-run analysis quarterly for new opportunities.")
        
        return "\n".join(steps)
    
    def generate_executive_summary_page(self, data):
        """Generate CEO/CXO-friendly executive summary page"""
        story = []
        
        # Main title
        story.append(Paragraph("Customer Recommendation Executive Report", self.styles['ExecutiveTitle']))
        story.append(Spacer(1, 20))
        
        # Customer snapshot
        customer_info = data.get('CustomerInfo', {})
        customer_classification = data.get('CustomerClassification', {})
        
        customer_name = customer_info.get('CustomerName', 'Unknown Customer')
        customer_id = customer_info.get('CustomerID', 'Unknown ID')
        number_of_stores = customer_classification.get('NumberOfStores', 0)
        total_quantity = customer_classification.get('TotalQuantitySold', 0)
        customer_type = customer_classification.get('CustomerType', 'Unknown')
        
        # Customer snapshot table (single row format)
        story.append(Paragraph("Customer Snapshot", self.styles['ExecutiveSubtitle']))
        
        customer_data = [
            ['Customer', 'Stores', 'Purchase Volume', 'Classification'],
            [f"{customer_name} ({customer_id})", f"{number_of_stores:,}", f"{total_quantity:,} units", customer_type]
        ]
        
        customer_table = Table(customer_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 2*inch])
        customer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D3D3D3')),  # Darker grey header
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('BACKGROUND', (0, 1), (-1, 1), colors.white),  # White background
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(customer_table)
        story.append(Spacer(1, 20))
        
        # Recommendation summary
        story.append(Paragraph("Recommendation Summary", self.styles['ExecutiveSubtitle']))
        
        # Calculate summary statistics
        accepted_recommendations = data.get('AcceptedRecommendations', [])
        rejected_recommendations = data.get('RejectedRecommendations', [])
        already_purchased_recommendations = data.get('AlreadyPurchasedRecommendations', [])
        
        total_accepted = sum(len(rec.get('CrossSell', [])) for rec in accepted_recommendations)
        total_rejected = sum(len(rec.get('RejectedCrossSell', [])) for rec in rejected_recommendations)
        total_already_purchased = sum(len(rec.get('AlreadyPurchasedCrossSell', [])) for rec in already_purchased_recommendations)
        
        # Count up-sell opportunities
        total_up_sell = 0
        for rec in accepted_recommendations:
            total_up_sell += len(rec.get('UpSell', []))
        
        summary_data = [
            ['Accepted Cross-Sell', 'Rejected', 'Already Purchased', 'Up-Sell'],
            [f"{total_accepted} items", f"{total_rejected} items", f"{total_already_purchased} item", f"{total_up_sell}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D3D3D3')),  # Darker grey header
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('BACKGROUND', (0, 1), (-1, 1), colors.white),  # White background
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Cross-sell recommendations table (top 7 for executive view)
        if total_accepted > 0:
            story.append(Paragraph("Cross-Sell Recommendations", self.styles['ExecutiveSubtitle']))
            
            # Prepare data for table (Base Product, Suggested Cross-Sell, Category, Price only)
            table_data = [['Base Product', 'Suggested Cross-Sell', 'Category', 'Price']]
            
            count = 0
            for rec in accepted_recommendations:
                if count >= 7:  # Limit to top 7 for executive summary
                    break
                cross_sell_items = rec.get('CrossSell', [])
                base_product = rec.get('ProductName', 'Unknown')
                
                for cross_sell in cross_sell_items:
                    if count >= 7:
                        break
                    product_name = cross_sell.get('SuggestedProduct', 'Unknown')
                    category = cross_sell.get('Category', 'Unknown')
                    price = cross_sell.get('Price', 0)
                    
                    # Show full values without truncation
                    table_data.append([base_product, product_name, category, f"${price}"])
                    count += 1
            
            # Create table with adjusted column widths - smaller Base Product, larger Suggested Cross-Sell
            cross_sell_table = Table(table_data, colWidths=[1.8*inch, 3.2*inch, 1.2*inch, 0.8*inch])
            cross_sell_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#87CEEB')),  # Sky blue header
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),  # White background
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (3, 1), (3, -1), 'CENTER'),  # Center align price column
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('WORDWRAP', (0, 1), (2, -1), 'CJK'),  # Enable word wrapping for text columns
                ('LEFTPADDING', (0, 1), (-1, -1), 6),
                ('RIGHTPADDING', (0, 1), (-1, -1), 6)
            ]))
            
            story.append(cross_sell_table)
            story.append(Spacer(1, 20))
        
        # Add page break before Executive Summary header
        story.append(PageBreak())
        
        # Executive Summary header on new page
        story.append(Paragraph("Executive Summary", self.styles['ExecutiveSubtitle']))
        story.append(Spacer(1, 15))
        
        # Generate Executive Summary using OpenAI GPT API
        print("🤖 Generating Executive Summary with OpenAI GPT API...")
        executive_summary_text = self.generate_executive_summary_with_api(data)
        
        # Add the API-generated summary
        story.append(Paragraph(executive_summary_text, self.styles['ExecutiveSummaryText']))
        story.append(Spacer(1, 20))
        
        # Add page break after executive summary
        story.append(PageBreak())
        
        return story
    
    def generate_customer_classification_analysis(self, customer_classification, customer_info=None):
        """Generate customer classification analysis section"""
        story = []
        
        # Customer name and ID header
        if customer_info:
            customer_name = customer_info.get('CustomerName', 'Unknown Customer')
            customer_id = customer_info.get('CustomerID', 'Unknown ID')
            customer_header = f"{customer_name} ({customer_id})"
            story.append(Paragraph(customer_header, self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
        
        # Section header
        story.append(Paragraph("Customer Classification Analysis", self.styles['SectionHeader']))
        
        # Customer type analysis
        customer_type = customer_classification.get('CustomerType', 'Unknown')
        total_quantity = customer_classification.get('TotalQuantitySold', 0)
        number_of_stores = customer_classification.get('NumberOfStores', 0)
        
        # Main analysis paragraph
        analysis_text = f"""
        The customer has been classified as a <b>{customer_type}</b> based on comprehensive analysis of their business operations. 
        This classification is determined through mathematical calculations considering both the scale of their store network and 
        their purchasing volume. The customer operates <b>{number_of_stores:,} stores</b> across their network and has purchased 
        a total of <b>{total_quantity:,} units</b> of products, indicating their market presence and purchasing power.
        """
        story.append(Paragraph(analysis_text, self.styles['CustomBodyText']))
        
        # Classification criteria analysis
        criteria = customer_classification.get('ClassificationCriteria', {})
        
        criteria_text = f"""
        The classification criteria analysis reveals that this customer meets the following thresholds: 
        {'✓' if criteria.get('StoresGreaterThan50', False) else '✗'} Stores > 50, 
        {'✓' if criteria.get('QuantityGreaterThan200K', False) else '✗'} Quantity > 200,000, 
        {'✓' if criteria.get('StoresBetween25And50', False) else '✗'} Stores between 25-50, 
        {'✓' if criteria.get('QuantityBetween50KAnd200K', False) else '✗'} Quantity between 50,000-200,000.
        """
        story.append(Paragraph(criteria_text, self.styles['Highlight']))
        
        # Business implications
        implications_text = f"""
        This classification has significant implications for our cross-selling strategy. As a {customer_type.lower()}, 
        the customer represents a {'high-value' if customer_type == 'CHG Own Sales Customer' else 'medium-value' if customer_type == 'Distributor Customer' else 'developing'} 
        business relationship that requires {'premium' if customer_type == 'CHG Own Sales Customer' else 'standard' if customer_type == 'Distributor Customer' else 'basic'} 
        attention and customized recommendations. The extensive store network and substantial purchasing volume suggest 
        strong potential for implementing comprehensive cross-selling initiatives across their operations.
        """
        story.append(Paragraph(implications_text, self.styles['CustomBodyText']))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_cross_sell_analysis(self, data):
        """Generate cross-sell analysis section with structured, attractive format"""
        story = []
        
        # Section header
        story.append(Paragraph("Cross-Sell Recommendations Analysis", self.styles['SectionHeader']))
        
        # Accepted recommendations
        accepted_recommendations = data.get('AcceptedRecommendations', [])
        if accepted_recommendations:
            story.append(Paragraph("Accepted Cross-Sell Opportunities", self.styles['SubsectionHeader']))
            
            # Summary box
            summary_text = f"✓ {len(accepted_recommendations)} catalogue items with validated cross-sell opportunities identified"
            story.append(Paragraph(summary_text, self.styles['Highlight']))
            story.append(Spacer(1, 15))
            
            # Create structured tables for each recommendation
            for rec in accepted_recommendations:
                item_name = rec.get('ProductName', 'Unknown Product')
                item_id = rec.get('CustomerCatalogueItemID', 'Unknown ID')
                cross_sell_items = rec.get('CrossSell', [])
                
                if cross_sell_items:
                    # Product header
                    story.append(Paragraph(f"<b>{item_name}</b> (ID: {item_id})", self.styles['CustomBodyText']))
                    story.append(Paragraph(f"Presents {len(cross_sell_items)} cross-sell opportunity{'ies' if len(cross_sell_items) > 1 else ''}:", self.styles['CustomBodyText']))
                    
                    # Create table for cross-sell items
                    table_data = [['Suggested Product', 'Product ID', 'Category', 'Price']]
                    
                    for cross_sell in cross_sell_items:
                        product_name = cross_sell.get('SuggestedProduct', 'Unknown Product')
                        product_id = cross_sell.get('ProductID', 'Unknown ID')
                        category = cross_sell.get('Category', 'Unknown Category')
                        price = cross_sell.get('Price', 0)
                        
                        # Truncate long product names
                        product_short = product_name[:40] + "..." if len(product_name) > 40 else product_name
                        
                        table_data.append([product_short, str(product_id), category, f"${price}"])
                    
                    # Create and style the table
                    cross_sell_table = Table(table_data, colWidths=[2.5*inch, 1.0*inch, 1.2*inch, 0.8*inch])
                    cross_sell_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8F5E8')),  # Light green header
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('ALIGN', (1, 1), (3, -1), 'CENTER'),  # Center align Product ID, Category, Price
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('WORDWRAP', (0, 1), (0, -1), 'CJK')
                    ]))
                    
                    story.append(cross_sell_table)
                    
                    # Add AI Reasoning below the table
                    story.append(Spacer(1, 10))
                    for cross_sell in cross_sell_items:
                        product_name = cross_sell.get('SuggestedProduct', 'Unknown Product')
                        reasoning = cross_sell.get('AIReasoning', 'No reasoning provided')
                        
                        # Truncate product name for display
                        product_short = product_name[:40] + "..." if len(product_name) > 40 else product_name
                        
                        reasoning_text = f"<b>{product_short}:</b> {reasoning}"
                        story.append(Paragraph(reasoning_text, self.styles['CustomBodyText']))
                        story.append(Spacer(1, 8))
                    
                    story.append(Spacer(1, 15))
        
        # Rejected recommendations
        rejected_recommendations = data.get('RejectedRecommendations', [])
        if rejected_recommendations:
            story.append(Paragraph("Rejected Cross-Sell Opportunities", self.styles['SubsectionHeader']))
            
            # Summary box
            summary_text = f"✗ {len(rejected_recommendations)} catalogue items with rejected cross-sell opportunities"
            story.append(Paragraph(summary_text, self.styles['Highlight']))
            story.append(Spacer(1, 15))
            
            # Create structured tables for rejected recommendations
            for rec in rejected_recommendations:
                item_name = rec.get('ProductName', 'Unknown Product')
                rejected_items = rec.get('RejectedCrossSell', [])
                
                if rejected_items:
                    # Product header
                    story.append(Paragraph(f"<b>{item_name}</b>", self.styles['CustomBodyText']))
                    story.append(Paragraph(f"Had {len(rejected_items)} potential cross-sell opportunity{'ies' if len(rejected_items) > 1 else ''} rejected:", self.styles['CustomBodyText']))
                    
                    # Create table for rejected items
                    table_data = [['Suggested Product', 'Product ID', 'Category', 'Price']]
                    
                    for rejected in rejected_items:
                        product_name = rejected.get('SuggestedProduct', 'Unknown Product')
                        product_id = rejected.get('ProductID', 'Unknown ID')
                        category = rejected.get('Category', 'Unknown Category')
                        price = rejected.get('Price', 0)
                        
                        # Truncate long product names
                        product_short = product_name[:40] + "..." if len(product_name) > 40 else product_name
                        
                        table_data.append([product_short, str(product_id), category, f"${price}"])
                    
                    # Create and style the table
                    rejected_table = Table(table_data, colWidths=[2.5*inch, 1.0*inch, 1.2*inch, 0.8*inch])
                    rejected_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFE8E8')),  # Light red header
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('ALIGN', (1, 1), (3, -1), 'CENTER'),  # Center align Product ID, Category, Price
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('WORDWRAP', (0, 1), (0, -1), 'CJK')
                    ]))
                    
                    story.append(rejected_table)
                    
                    # Add AI Reasoning below the table
                    story.append(Spacer(1, 10))
                    for rejected in rejected_items:
                        product_name = rejected.get('SuggestedProduct', 'Unknown Product')
                        reasoning = rejected.get('AIReasoning', 'No reasoning provided')
                        
                        # Truncate product name for display
                        product_short = product_name[:40] + "..." if len(product_name) > 40 else product_name
                        
                        reasoning_text = f"<b>{product_short}:</b> {reasoning}"
                        story.append(Paragraph(reasoning_text, self.styles['CustomBodyText']))
                        story.append(Spacer(1, 8))
                    
                    story.append(Spacer(1, 15))
        
        # Already purchased recommendations
        already_purchased_recommendations = data.get('AlreadyPurchasedRecommendations', [])
        if already_purchased_recommendations:
            story.append(Paragraph("Already Purchased Cross-Sell Opportunities", self.styles['SubsectionHeader']))
            
            # Summary box
            summary_text = f"✓ {len(already_purchased_recommendations)} catalogue items with products already purchased"
            story.append(Paragraph(summary_text, self.styles['Highlight']))
            story.append(Spacer(1, 15))
            
            # Create structured tables for already purchased recommendations
            for rec in already_purchased_recommendations:
                item_name = rec.get('ProductName', 'Unknown Product')
                purchased_items = rec.get('AlreadyPurchasedCrossSell', [])
                
                if purchased_items:
                    # Product header
                    story.append(Paragraph(f"<b>{item_name}</b>", self.styles['CustomBodyText']))
                    story.append(Paragraph(f"Has {len(purchased_items)} cross-sell product{'s' if len(purchased_items) > 1 else ''} already purchased:", self.styles['CustomBodyText']))
                    
                    # Create table for purchased items
                    table_data = [['Suggested Product', 'Product ID', 'Category', 'Price']]
                    
                    for purchased in purchased_items:
                        product_name = purchased.get('SuggestedProduct', 'Unknown Product')
                        product_id = purchased.get('ProductID', 'Unknown ID')
                        category = purchased.get('Category', 'Unknown Category')
                        price = purchased.get('Price', 0)
                        
                        # Truncate long product names
                        product_short = product_name[:40] + "..." if len(product_name) > 40 else product_name
                        
                        table_data.append([product_short, str(product_id), category, f"${price}"])
                    
                    # Create and style the table
                    purchased_table = Table(table_data, colWidths=[2.5*inch, 1.0*inch, 1.2*inch, 0.8*inch])
                    purchased_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8F4FD')),  # Light blue header
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('ALIGN', (1, 1), (3, -1), 'CENTER'),  # Center align Product ID, Category, Price
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('WORDWRAP', (0, 1), (0, -1), 'CJK')
                    ]))
                    
                    story.append(purchased_table)
                    
                    # Add AI Reasoning below the table
                    story.append(Spacer(1, 10))
                    for purchased in purchased_items:
                        product_name = purchased.get('SuggestedProduct', 'Unknown Product')
                        reasoning = purchased.get('AIReasoning', 'No reasoning provided')
                        
                        # Truncate product name for display
                        product_short = product_name[:40] + "..." if len(product_name) > 40 else product_name
                        
                        reasoning_text = f"<b>{product_short}:</b> {reasoning}"
                        story.append(Paragraph(reasoning_text, self.styles['CustomBodyText']))
                        story.append(Spacer(1, 8))
                    
                    story.append(Spacer(1, 15))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_up_sell_analysis(self, data):
        """Generate up-sell analysis section"""
        story = []
        
        # Section header
        story.append(Paragraph("Up-Sell Recommendations Analysis", self.styles['SectionHeader']))
        
        accepted_recommendations = data.get('AcceptedRecommendations', [])
        up_sell_count = 0
        
        for rec in accepted_recommendations:
            up_sell_items = rec.get('UpSell', [])
            up_sell_count += len(up_sell_items)
        
        if up_sell_count > 0:
            up_sell_text = f"""
            The analysis identified <b>{up_sell_count} up-sell opportunities</b> across the customer's catalogue items. 
            These recommendations focus on encouraging the customer to purchase higher-value versions of products 
            they already use, or to increase their order quantities for existing products. Up-sell opportunities 
            are carefully evaluated to ensure they provide genuine value to the customer while supporting business growth.
            """
            story.append(Paragraph(up_sell_text, self.styles['CustomBodyText']))
        else:
            up_sell_text = """
            <b>No up-sell opportunities were identified</b> in the current analysis. This may indicate that 
            the customer is already purchasing optimal quantities of products, or that the current catalogue 
            items don't present clear up-sell scenarios. The system continues to monitor for future up-sell 
            opportunities as the customer's needs evolve.
            """
            story.append(Paragraph(up_sell_text, self.styles['CustomBodyText']))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_summary_analysis(self, data):
        """Generate summary analysis section"""
        story = []
        
        # Section header
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        summary = data.get('Summary', {})
        total_up_sell = summary.get('TotalUpSell', 0)
        total_cross_sell = summary.get('TotalCrossSell', 0)
        total_rejected = summary.get('TotalRejected', 0)
        total_already_purchased = summary.get('TotalAlreadyPurchased', 0)
        total_recommendations = summary.get('TotalRecommendations', 0)
        
        # Overall analysis
        summary_text = f"""
        The comprehensive analysis of the customer's catalogue and purchase history has yielded significant 
        insights for cross-selling strategy development. The system processed multiple catalogue items and 
        identified <b>{total_recommendations} viable recommendations</b> across up-sell and cross-sell categories.
        """
        story.append(Paragraph(summary_text, self.styles['CustomBodyText']))
        
        # Recommendation breakdown
        breakdown_text = f"""
        The recommendation breakdown shows <b>{total_cross_sell} accepted cross-sell opportunities</b>, 
        <b>{total_rejected} rejected opportunities</b> (demonstrating quality control), and 
        <b>{total_already_purchased} already purchased items</b> (showing effective duplicate prevention). 
        Additionally, <b>{total_up_sell} up-sell opportunities</b> were identified to enhance customer value.
        """
        story.append(Paragraph(breakdown_text, self.styles['CustomBodyText']))
        
        # Strategic implications
        implications_text = f"""
        These findings provide a solid foundation for implementing targeted cross-selling initiatives. 
        The high number of accepted recommendations indicates strong potential for revenue growth, while 
        the rejection rate demonstrates the system's commitment to quality and accuracy. The identification 
        of already purchased items shows effective inventory tracking and prevents redundant recommendations.
        """
        story.append(Paragraph(implications_text, self.styles['CustomBodyText']))
        
        # Next steps
        next_steps_text = f"""
        <b>Recommended Next Steps:</b> Focus on the {total_cross_sell} accepted cross-sell opportunities 
        as priority implementation targets. These recommendations have been validated through AI analysis 
        and represent the highest probability of successful adoption by the customer. Regular monitoring 
        of these recommendations will help track implementation success and identify additional opportunities.
        """
        story.append(Paragraph(next_steps_text, self.styles['Highlight']))
        
        story.append(Spacer(1, 12))
        return story
    
    def generate_pdf_report(self, json_file_path, output_pdf_path=None):
        """Generate comprehensive PDF report from JSON data"""
        # Load JSON data
        data = self.load_json_data(json_file_path)
        if not data:
            return False
        
        # Generate output filename if not provided
        if not output_pdf_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_pdf_path = f"reports/analysis_report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(output_pdf_path, pagesize=A4, rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        
        story = []
        
        # Executive Summary Page (CEO/CXO-friendly)
        story.extend(self.generate_executive_summary_page(data))
        
        # Cross-Sell Analysis
        story.extend(self.generate_cross_sell_analysis(data))
        
        # Up-Sell Analysis
        story.extend(self.generate_up_sell_analysis(data))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"✅ PDF report generated successfully: {output_pdf_path}")
            return True
        except Exception as e:
            print(f"❌ Error generating PDF report: {str(e)}")
            return False

def main():
    """Main function to generate PDF report"""
    generator = PDFReportGenerator()
    
    # Find the most recent JSON file
    json_files = [f for f in os.listdir('recommendations') if f.startswith('recommendations_') and f.endswith('.json')]
    
    if not json_files:
        print("❌ No recommendation JSON files found in current directory")
        return
    
    # Sort by modification time to get the most recent
    json_files.sort(key=lambda x: os.path.getmtime(os.path.join('recommendations', x)), reverse=True)
    latest_json_file = os.path.join('recommendations', json_files[0])
    
    print(f"📄 Found JSON file: {latest_json_file}")
    
    # Generate PDF report
    success = generator.generate_pdf_report(latest_json_file)
    
    if success:
        print("🎉 PDF analysis report generated successfully!")
    else:
        print("❌ Failed to generate PDF report")

if __name__ == "__main__":
    main() 