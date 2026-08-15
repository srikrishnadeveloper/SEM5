// Online Java Compiler
// Use this editor to write, compile and run your Java code online
import java.util.Scanner;

class q1 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in)
        String username = "krishna"
        String password = "krishna123"
        // first four tires:
        for(int attempts=4;attempts>1;attempts--){
            System.out.println("Login system");
            System.out.println("Enter username");
            String lusername = sc.nextLine();
            System.out.println("Enter password");
            String lpassword = sc.nextLine();
            if(rusername==lusername & rpassword==lpassword){
                System.out.println("you are logged");
                break;
            }
            System.out.println("you have still"+attempts+"attempts");
        }
        System.out.println("you are blocked from your account");
        sc.close();

    }
}
