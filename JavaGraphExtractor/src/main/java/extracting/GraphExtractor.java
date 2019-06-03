package extracting;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.eclipse.jdt.core.dom.ASTNode;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.json.JSONArray;
import org.json.JSONObject;

import misc.Util;
import parsing.ASTToken;
import parsing.JavaLexer;
import parsing.JavaParser;

public class GraphExtractor {
	public static void main(String[] args) {
		if (args.length == 0) {
			System.out.println("Provide at least one argument (input-directory)");
			System.exit(1);
		}
		File projectDir = new File(args[0]);
		List<JSONObject> graphs = collectMethodGraphs(projectDir);
		writeGraphs(args, projectDir, graphs);
	}

	private static void writeGraphs(String[] args, File projectDir, List<JSONObject> graphs) {
		String text = new JSONArray(graphs).toString();
		File outFile = new File(args.length > 1 ? args[1] : projectDir.getParentFile() + "/output.json");
		try (BufferedWriter fw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outFile), StandardCharsets.UTF_8))) {
			fw.write(text);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static List<JSONObject> collectMethodGraphs(File projectDir) {
		List<File> files = Util.collectFiles(projectDir);
		JavaParser parser = new JavaParser(projectDir);
		List<JSONObject> graphs = new ArrayList<>();
		for (File f : files) {
			try {
				String text = Util.readLines(f).stream().collect(Collectors.joining("\n"));
				CompilationUnit cu = parser.parse(text);
				ASTToken root = parser.processCU(text, cu);
				Map<Integer, String> comments = parser.getComments(text, cu);
				for (ASTToken method : traverse(root)) {
					graphs.add(extract(f, method, comments));
				}
			} catch (Exception e) {
				continue;
			}
		}
		return graphs;
	}
	
	private static List<ASTToken> traverse(ASTToken root) {
		List<ASTToken> methods = new ArrayList<>();
		if (root.getText().equals("#31")) {
			getFullName(root);
			methods.add(root);
		}
		else if (root.hasChildren()) {
			for (ASTToken child : root.getChildren()) methods.addAll(traverse(child));
		}
		return methods;
	}
	
	private static JSONObject extract(File file, ASTToken method, Map<Integer, String> comments) {
		Map<Integer, Integer> parents = new HashMap<>();
		Map<Integer, String> tokens = new LinkedHashMap<>();
		extractTokens(method, 1, tokens, parents);
		JSONObject graph = new JSONObject();
		graph.put("FileName", file.getAbsolutePath());
		graph.put("MethodName", getFullName(method));
		JSONObject labels = new JSONObject();
		for (int i = 0; i < tokens.size(); i++) {
			labels.put(Integer.toString(i), tokens.get(i));
		}
		graph.put("Nodes", labels);
		
		graph.put("Comments", alignComments(method, comments));
		
		// Collect edges
		JSONObject edges = new JSONObject();
		// Next token edges
		JSONArray nextToken = new JSONArray();
		Map<Integer, Integer> nextTokens = new HashMap<>();
		int prev = -1;
		for (int ix : tokens.keySet()) {
			if (parents.values().stream().anyMatch(p -> p == ix)) continue;
			if (prev != -1) {
				JSONArray edge = new JSONArray();
				edge.put(prev);
				edge.put(ix);
				nextToken.put(edge);
				nextTokens.put(prev, ix);
			}
			prev = ix;
		}
		edges.put("NextToken", nextToken);
		
		// Last use edges
		JSONArray lastLexicalUse = new JSONArray();
		Map<String, Integer> lastLocs = new HashMap<>();
		for (Entry<Integer, String> entry : tokens.entrySet()) {
			String name = entry.getValue();
			if (JavaLexer.isID(name)) {
				if (lastLocs.containsKey(name)) {
					JSONArray edge = new JSONArray();
					edge.put(entry.getKey());
					edge.put(lastLocs.get(name));
					lastLexicalUse.put(edge);
				}
				lastLocs.merge(name, entry.getKey(), Integer::max);
			}
		}
		edges.put("LastLexicalUse", lastLexicalUse);
		
		// Child edges
		JSONArray child = new JSONArray();
		for (Entry<Integer, Integer> entry : parents.entrySet()) {
			JSONArray edge = new JSONArray();
			edge.put(entry.getValue());
			edge.put(entry.getKey());
			child.put(edge);
		}
		edges.put("Child", child);
		
		// Store edges
		graph.put("Edges", edges);
		return graph;
	}
	
	private static Function<ASTToken, Stream<ASTToken>> flattenTree =
			t -> !t.hasChildren() ? Stream.of(t)
					: Stream.concat(Stream.of(t), t.getChildren().stream().flatMap(c -> GraphExtractor.flattenTree.apply(c)));
	private static Map<Integer, String> alignComments(ASTToken method, Map<Integer, String> comments) {
		Map<Integer, Integer> commentLocs = new HashMap<>(); // First, we track the alignment of comment-index to node index
		List<ASTToken> flattened = flattenTree.apply(method).collect(Collectors.toList());
		for (int i = 0; i < flattened.size(); i++) {
			for (Entry<Integer, String> e : comments.entrySet()) {
				// Align each comment with the last node that is on the same line as that comment (if any)
				int commentLine = e.getKey();
				if (commentLine < method.getLine()) continue;
				// Heuristically, align with the first (highest in the AST) node that starts on this line
				if (commentLine <= flattened.get(i).getLine() && !commentLocs.containsKey(e.getKey())) {
					commentLocs.put(e.getKey(), i); // We simply store the comment's line number, because there can only be one comment per line
				}
			}
		}
		// Convert location alignment to map of <nearest-node-index, comment-text>
		return commentLocs.entrySet().stream().collect(Collectors.toMap(e -> e.getValue(), e -> comments.get(e.getKey())));
	}

	private static int extractTokens(ASTToken curr, int ix, Map<Integer, String> tokens, Map<Integer, Integer> parents) {
		tokens.put(ix, name(curr));
		int currIx = ix;
		if (!curr.hasChildren()) return currIx;
		for (int i = 0; i < curr.getChildren().size(); i++) {
			ASTToken child = curr.getChild(i);
			currIx += 1;
			parents.put(currIx, ix);
			currIx = extractTokens(child, currIx, tokens, parents);
		}
		return currIx;
	}

	private static String name(ASTToken token) {
		if (token.getText().startsWith("#")) {
			String className = ASTNode.nodeClassForType(Integer.parseInt(token.getText().substring(1))).getName();
			return className.contains(".") ? className.substring(className.lastIndexOf(".") + 1) : className;
		}
		else return token.getText();
	}

	private static String getFullName(ASTToken root) {
		String name = "";
		String parameters = "";
		for (ASTToken child : root.getChildren()) {
			if (child.getText().equals("#42")) name = child.getChild(0).getText();
			else if (child.getText().equals("#44")) {
				if (!parameters.isEmpty()) parameters += ", ";
				String type = GraphExtractor.flattenTree.apply(child.getChild(0))
						.map(n -> n.getText())
						.filter(t -> !t.startsWith("#"))
						.collect(Collectors.joining(""));
				type += " ";
				type += GraphExtractor.flattenTree.apply(child.getChild(1))
						.map(n -> n.getText())
						.filter(t -> !t.startsWith("#"))
						.collect(Collectors.joining(""));
				parameters += type;
			}
		}
		String className = "";
		for (ASTToken child : root.getParent().getChildren()) {
			if (child.getText().equals("#42")) className = child.getChild(0).getText();
		}
		return className + "." + name + "(" + parameters + ")";
	}
}
