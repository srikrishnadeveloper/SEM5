// Online Java Compiler
// Use this editor to write, compile and run your Java code online
import java.util.Scanner;

class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String UserName = "krishna";
        String CardPin = "123456";
        int amount = 2000;
        
        for(int attempts=4;attempts>1;attempts--){
            System.out.println("ATM SYSTEM");
            System.out.println("Enter UserName");
            String CUserName = sc.nextLine();
            System.out.println("Enter CardPin");
            String CCardPin = sc.nextLine();
            if(UserName==CUserName & CardPin==CCardPin){
                System.out.println("current balance"+amount);
                
                System.out.println("Enter action:1=deposit & 2 = withdraw");
                int action = sc.nextInt();
                if(action==1){
                    System.out.println("Enter the amount that should be deposited");
                    int deposit = sc.nextInt();
                    amount += deposit;
                    System.out.println("Your new balace:" + amount);
                }else if(action==2){
                    System.out.println("Enter amount to withdraw");
                    int withdraw = sc.nextInt();
                    if(amount>withdraw){
                        amount -=withdraw;
                        System.out.println("Your new balace:" + amount);
                    }else{
                        System.out.println("Enter valid amount,Enter amount should be less than your balance");
                    }
                }else{
                    System.out.println("Enter Valid action");
                }
                
                
            }
            System.out.println("you have still"+attempts+"attempts");
        }
        System.out.println("you are blocked from your account");
        sc.close();

    }
}
