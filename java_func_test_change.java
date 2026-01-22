
import com.sun.jarsigner.ContentSignerParameters;

public class ContentSignerUtil {

    public static void printSignerDetails(ContentSignerParameters parameters) {
        if (parameters == null) {
            throw new IllegalArgumentException("ContentSignerParameters must not be null");
        }

        String algorithm = parameters.getSignatureAlgorithm();
        byte[] signature = parameters.getSignature();

        System.out.println("Signature Algorithm: " + algorithm);
        System.out.println("Signature: " + new String(signature));
    }
}
