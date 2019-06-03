package misc;



import org.eclipse.jdt.core.dom.ASTNode;

/**
 *  Helper file to print either the full or simplified type
 *  of the Eclipse AST node.
 */
public class ASTNodeTypePrinter {
	
	/**
	 * Helper method to remove the package structure from the type name.
	 * @param node
	 * @return
	 */
	public static String getSimpleType(ASTNode node)
	{
		String fullType = ASTNode.nodeClassForType(node.getNodeType()).getName();
		String[] pieces = fullType.split("\\.");
		return pieces[pieces.length -1];
	}
	
	/**
	 * Returns a string with the fully qualified type.
	 * @param node
	 * @return
	 */
	public static String getType(ASTNode node)
	{
		return ASTNode.nodeClassForType(node.getNodeType()).getName();
	}

}