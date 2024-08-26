class LoanCalculator:
    def __init__(self, principal, interest_rate, duration_years):
        # Initialize the LoanCalculator object with the provided parameters
        self.principal = principal
        # Convert annual interest rate from percentage to decimal
        self.interest_rate = interest_rate / 100
        # Save the loan duration in years
        self.duration_years = duration_years

    def calculate_monthly_payment(self):
        # Calculate the monthly interest rate by dividing the annual rate by 12
        monthly_interest_rate = self.interest_rate / 12
        # Calculate the total number of payments over the loan term
        num_payments = self.duration_years * 12
        # Calculate the monthly payment using the formula for fixed-rate loans
        monthly_payment = (self.principal * monthly_interest_rate) / (1 - (1 + monthly_interest_rate) ** -num_payments)
        return monthly_payment

    def calculate_total_interest(self):
        # Calculate the monthly payment for the loan
        monthly_payment = self.calculate_monthly_payment()
        # Calculate the total number of payments over the loan term
        num_payments = self.duration_years * 12
        # Calculate the total interest paid over the loan term
        total_interest = (monthly_payment * num_payments) - self.principal
        return total_interest


# Example usage:
if __name__ == "__main__":
    # Prompt the user to input the principal amount, annual interest rate, and loan duration
    principal = float(input("Enter loan amount: "))
    interest_rate = float(input("Enter annual interest rate (%): "))
    duration_years = int(input("Enter loan duration in years: "))

    # Create a LoanCalculator object with the provided inputs
    loan_calculator = LoanCalculator(principal, interest_rate, duration_years)
    
    # Calculate the monthly payment and total interest using the LoanCalculator object
    monthly_payment = loan_calculator.calculate_monthly_payment()
    total_interest = loan_calculator.calculate_total_interest()

    # Print the calculated monthly payment and total interest
    print(f"Monthly payment: ${monthly_payment:.2f}")
    print(f"Total interest: ${total_interest:.2f}")