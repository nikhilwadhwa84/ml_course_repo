package parsing;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.eclipse.jdt.core.dom.ASTNode;

class ASTNodeWrapper {
	final ASTNode node;
	final List<ASTNodeWrapper> children;
	
	ASTNodeWrapper(ASTNode node) {
		this.node = node;
		this.children = new ArrayList<>();
	}
	
	void addChild(ASTNodeWrapper node) {
		if (node != null) this.children.add(node);
	}

	public ASTNode getNode() {
		return node;
	}

	public List<ASTNodeWrapper> getChildren() {
		return children;
	}
	
	@Override
	public String toString() {
		return this.node.getNodeType() +
				" [" + this.children.stream().map(node -> node.node.getNodeType()+"").collect(Collectors.joining(", ")) + "]";
	}
}