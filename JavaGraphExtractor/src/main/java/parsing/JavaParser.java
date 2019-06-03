package parsing;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.eclipse.jdt.core.JavaCore;
import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.ASTVisitor;
import org.eclipse.jdt.core.dom.BlockComment;
import org.eclipse.jdt.core.dom.Comment;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.IBinding;
import org.eclipse.jdt.core.dom.IMethodBinding;
import org.eclipse.jdt.core.dom.ITypeBinding;
import org.eclipse.jdt.core.dom.IVariableBinding;
import org.eclipse.jdt.core.dom.LineComment;
import org.eclipse.jdt.core.dom.NumberLiteral;
import org.eclipse.jdt.core.dom.SimpleName;
import org.eclipse.jdt.core.dom.StructuralPropertyDescriptor;

public class JavaParser {

	private ASTParser parser;
	private CompilationUnit currUnit;
	private String[] classpath;

	public JavaParser() {
	}

	public JavaParser(File projectRoot) {
		this.parser = ASTParser.newParser(AST.JLS11);
		Map<String, String> options = JavaCore.getOptions();
		options.put(JavaCore.COMPILER_SOURCE, JavaCore.VERSION_1_8);
		this.parser.setCompilerOptions(options);
		this.parser.setKind(ASTParser.K_COMPILATION_UNIT);
	}

	public static List<List<String>> traverse(ASTToken token) {
		List<List<String>> tokens = new ArrayList<>();
		traverse(token, tokens);
		return tokens;
	}

	private static void traverse(ASTToken token, List<List<String>> tokens) {
		int line = token.getLine();
		while (tokens.size() <= line)
			tokens.add(new ArrayList<>());
		if (token.isTerminal())
			tokens.get(line).add(token.getText() + (token.getType().isEmpty() ? "" : ":" + token.getType()));
		else if (token.hasChildren())
			for (ASTToken child : token.getChildren())
				traverse(child, tokens);
	}
	
	public CompilationUnit parse(String text) throws IOException {
		this.parser.setResolveBindings(true);
		this.parser.setBindingsRecovery(true);
		this.parser.setEnvironment(this.classpath, new String[] { "" }, new String[] { "UTF-8" }, true);
		this.parser.setSource(text.toCharArray());
		CompilationUnit cu = (CompilationUnit) this.parser.createAST(null);
		this.currUnit = cu;
		return cu;
	}
	
	public Map<Integer, String> getComments(String text, CompilationUnit cu) {
		Map<Integer, String> comments = new HashMap<>();
		for (Object x : cu.getCommentList()) {
			if (x instanceof LineComment) {
				LineComment lc = (LineComment) x;
				String comment = text.substring(lc.getStartPosition() + 2, lc.getStartPosition() + lc.getLength()).trim();
				comments.put(cu.getLineNumber(lc.getStartPosition()), comment);
			}
		}
		return comments;
	}

	public ASTToken processCU(String text, ASTNode node) {
		ASTNodeWrapper nodeTree = visit(node);
		fixArrayDimensions(nodeTree);
		return collectNodes(text, nodeTree, null);
	}

	@SuppressWarnings("rawtypes")
	private ASTNodeWrapper visit(ASTNode node) throws IllegalArgumentException {
		// Tentatively, skip JavaDoc
		if (node.getNodeType() == 29)
			return null;
		ASTNodeWrapper wrapper = new ASTNodeWrapper(node);
		List list = node.structuralPropertiesForType();
		
		//CompilationUnit cunit = (CompilationUnit)node;
		//List comments = cunit.getCommentList();
		//SimplePropertyDescriptor commentSpd = new SimplePropertyDescriptor();
		//node.setProperty("comments", comments);
		//System.out.println(comments);
		
		
		// For all child nodes ...
		for (int i = 0; i < list.size(); i++) {
			// ... retrieve the associated property,
			StructuralPropertyDescriptor curr = (StructuralPropertyDescriptor) list.get(i);
			Object child = node.getStructuralProperty(curr);
			// if the child is a regular AST node, visit it directly,
			if (child instanceof ASTNode) {
				wrapper.addChild(visit((ASTNode) child));
			}
			// else if the child is a list of nodes, flatten and visit in order,
			else if (child instanceof List) {
				List children = (List) child;
				for (Object el : children) {
					if (el instanceof ASTNode) {
						wrapper.addChild(visit((ASTNode) el));
					}
				}
			}
		}
		return wrapper;
	}

	/**
	 * For some reason, the dimensions of an array are added as a list to the end of
	 * ArrayCreation nodes (node-type 3) instead of as a child to their
	 * corresponding dimension nodes. This code fixes that.
	 */
	private void fixArrayDimensions(ASTNodeWrapper nodeTree) {
		for (ASTNodeWrapper child : nodeTree.getChildren())
			fixArrayDimensions(child);
		if (nodeTree.getNode().getNodeType() == 3) {
			List<ASTNodeWrapper> children = nodeTree.getChildren();
			List<ASTNodeWrapper> typeChildren = children.get(0).getChildren();
			for (int i = children.size() - 1; i > 0; i--) {
				if (children.get(i).getNode().getNodeType() == 4)
					continue;
				ASTNodeWrapper node = children.remove(i);
				typeChildren.get(i).addChild(node);
			}
		}
	}

	private ASTToken collectNodes(String text, ASTNodeWrapper tree, ASTToken parent) {
		ASTNode root = tree.node;
		int start = root.getStartPosition();
		int end = start + root.getLength();
		if (root.getLength() > 1000000000) {
			throw new IllegalArgumentException(
					"Syntax error in current file near node at " + start + ", content: " + root.toString());
		}
		ASTToken currToken = addCurrent(parent, tree);
		for (ASTNodeWrapper childTree : tree.children) {
			ASTNode childNode = childTree.node;
			int childIndex = childNode.getStartPosition();
			// Add any punctuation and similar tokens in between child-nodes
			if (childIndex > start) {
				getTerminals(text.substring(start, childIndex), start, currToken).forEach(currToken::addChild);
				start = childIndex;
			}
			currToken.addChild(collectNodes(text, childTree, currToken));
			start += childNode.getLength();
			if (start != childNode.getLength() + childIndex) {
				System.err.println("Position mismatch: " + start + ", " + childIndex + ", " + childNode.getLength());
				start = childIndex + childNode.getLength();
			}
		}
		if (start < end)
			getTerminals(text.substring(start, end), start, currToken).forEach(currToken::addChild);
		return currToken;
	}
	
	private ASTToken addCurrent(ASTToken parent, ASTNodeWrapper tree) {
		ASTNode root = tree.node;
		String value = "#" + Integer.toString(root.getNodeType());
		String type = extractType(root);
		int startLine = this.currUnit == null ? 0 : this.currUnit.getLineNumber(root.getStartPosition());
		ASTToken currToken = new ASTToken(parent, value, startLine, type);
		return currToken;
	}
	
	private String extractType(ASTNode root) {
		if (root.getNodeType() == 42) {
			SimpleName name = (SimpleName) root;
			ITypeBinding typeBinding = name.resolveTypeBinding();
			// Retrieve binding indirectly if necessary
			IBinding binding;
			if (typeBinding == null && (binding = name.resolveBinding()) != null) {
				if (binding.getKind() == IBinding.VARIABLE) {
					IVariableBinding varBinding = (IVariableBinding) binding;
					typeBinding = varBinding.getType();
				} else if (binding.getKind() == IBinding.METHOD) {
					IMethodBinding methodBinding = (IMethodBinding) binding;
					typeBinding = methodBinding.getReturnType();
				}
			}
			if (typeBinding != null) {
				return typeBinding.getQualifiedName();
			}
		} else if (root.getNodeType() == 9)
			return "boolean";
		else if (root.getNodeType() == 13)
			return "char";
		else if (root.getNodeType() == 45)
			return "java.lang.String";
		else if (root.getNodeType() == 34) {
			NumberLiteral nl = (NumberLiteral) root;
			ITypeBinding typeBinding = nl.resolveTypeBinding();
			if (typeBinding != null)
				return typeBinding.getName();
		}
		return "";
	}

	private List<ASTToken> getTerminals(String intermediate, int index, ASTToken token) {
		String content = intermediate.trim().replaceAll("[\n\r]+", "");
		if (!content.isEmpty()) {
			List<String> terminals = new JavaLexer().lex(intermediate).stream().flatMap(l -> l.stream()).collect(Collectors.toList());
			List<ASTToken> tokens = new ArrayList<>();
			for (int i = 0; i < terminals.size(); i++) {
				tokens.add(new ASTToken(token, terminals.get(i), -1));
			}
			return tokens;
		}
		return Collections.emptyList();
	}
}

class CommentVisitor extends ASTVisitor {
	CompilationUnit cu;
	String source;
 
	public CommentVisitor(CompilationUnit cu, String source) {
		super();
		this.cu = cu;
		this.source = source;
	}
 
	public boolean visit(LineComment node) {
		int start = node.getStartPosition();
		int end = start + node.getLength();
		String comment = source.substring(start, end);
		System.out.println(comment);
		return true;
	}
 
	public boolean visit(BlockComment node) {
		int start = node.getStartPosition();
		int end = start + node.getLength();
		String comment = source.substring(start, end);
		System.out.println(comment);
		return true;
	}
 
}
