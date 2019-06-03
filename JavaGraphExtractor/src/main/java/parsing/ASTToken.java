package parsing;

import java.util.ArrayList;
import java.util.List;

public class ASTToken {

	private ASTToken parent;
	private List<ASTToken> children;
	
	private final String text;
	private final String type;
	private int line;
	
	public ASTToken(ASTToken parent, String text, int line) {
		this(parent, text, line, "");
	}

	public ASTToken(ASTToken parent, String text, int line, String type) {
		this.parent = parent;
		this.text = text;
		this.line = line;
		this.type = type;
		this.children = null;
	}
	
	public ASTToken getParent() {
		return this.parent;
	}

	/**
	 * A node is terminal iff it has no children <b>AND</b> it is not marked with a #!
	 * The # prefix identifies designated non-terminal nodes, which can have no children in rare cases
	 * @return
	 */
	public boolean isTerminal() {
		return !this.hasChildren() && this.text.charAt(0) != '#';
	}
	
	/**
	 * Confirms that this node has child-nodes.
	 * <br/>
	 * <b>NOTE:</b> having no children is <u>not</u> equivalent to being a terminal!
	 * Use {@link #isTerminal()} to check that.
	 * The other way around does hold: having children --> being non-terminal
	 * @return
	 */
	public boolean hasChildren() {
		return this.children != null && !this.children.isEmpty();
	}
	
	public List<ASTToken> getChildren() {
		return this.children;
	}

	public ASTToken getChild(int index) {
		return this.getChildren().get(index);
	}

	public String getText() {
		return this.text;
	}

	public int getLine() {
		return this.line;
	}
	
	public String getType() {
		return type;
	}

	public void addChild(ASTToken token) {
		if (this.children == null) {
			this.children = new ArrayList<ASTToken>();
		}
		this.children.add(token);
	}

	public void addChild(int index, ASTToken token) {
		if (this.children == null) {
			this.children = new ArrayList<ASTToken>();
		}
		this.children.add(index, token);
	}
	
	public ASTToken removeChild(ASTToken token) {
		if (this.children == null) return null;
		this.children.remove(token);
		token.parent = null;
		return token;
	}
	
	@Override
	public boolean equals(Object other) {
		if (other == this) return true;
		if (other == null || !(other instanceof ASTToken)) return false;
		ASTToken token = (ASTToken) other;
		return equalsToken(token);
	}
	
	private boolean equalsToken(ASTToken token) {
		if (!this.text.equals(token.text)) return false;
		if (!this.type.equals(token.type)) return false;
		if (this.hasChildren() != token.hasChildren()) return false;
		if (this.hasChildren()) return this.children.equals(token.children);
		return true;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.text);
		sb.append("\t");
		sb.append(this.line);
		if (!this.type.isEmpty()) {
			sb.append("\t");
			sb.append(this.type);
		}
		return sb.toString();
	}
}
